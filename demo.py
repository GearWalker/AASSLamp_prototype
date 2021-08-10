import numpy as np
from collections import namedtuple
import mediapipe_utils as mpu
import depthai as dai
import cv2
from scipy.interpolate import griddata
from pathlib import Path

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape, interpolation=cv2.INTER_NEAREST)
    #print(f'in  = {resized.shape}')
    resized = resized.transpose(2,0,1)
    #print(f'out = {resized.shape}')
    return resized

class Marker:
    
    def __init__(self):
        # Fixed size
        self.markerCoordinates = [dai.Point3f(0,0,0) for i in range(4)] # fixed size to 4 (markers max)
        self.fingerCoordinates = [] # variable size (multitouch supported)

        self.preview_width = 640
        self.preview_height = 400

        self.playground_width = 640
        self.playground_height = 400

        self.touch_treshold = 30 # mm above playground surface 

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()

        self.pd_path="models/palm_detection_6_shaves.blob"
        self.pd_score_thresh=0.65
        self.pd_nms_thresh=0.3
        self.pd_input_length = 128
        self.lm_path="models/hand_landmark_6_shaves.blob"
        self.lm_score_threshold=0.5

        self.show_finger_index = True
        self.show_hand_landmarks = False
        self.show_hand_box = True

        self.roi_size = 5.0

        anchor_options = mpu.SSDAnchorOptions(num_layers=4, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=128,
                                input_size_width=128,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 16, 16],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
        
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

    def create_pipeline(self):
        print("Creating pipeline...")
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.initialControl.setManualFocus(130)
        cam.initialControl.AntiBandingMode(dai.CameraControl.AntiBandingMode.MAINS_50_HZ)
        cam.setPreviewSize(self.preview_width, self.preview_height)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        print("Creating Left Mono Camera...")
        left = pipeline.createMonoCamera()
        left.initialControl.setManualFocus(130)
        left.initialControl.AntiBandingMode(dai.CameraControl.AntiBandingMode.MAINS_60_HZ)
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        print("Creating Right Mono Camera...")
        right = pipeline.createMonoCamera()
        right.initialControl.setManualFocus(130)
        right.initialControl.AntiBandingMode(dai.CameraControl.AntiBandingMode.MAINS_60_HZ)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        print("Creating Stereo Depth Map...")
        depth = pipeline.createStereoDepth()
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.initialConfig.setConfidenceThreshold(230)
        depth.setDepthAlign(dai.CameraBoardSocket.RGB)
        depth.setLeftRightCheck(True)
        depth.setExtendedDisparity(False)
        depth.setSubpixel(False)
        depth.setOutputSize(self.preview_width, self.preview_height)

        # connect pipeline
        left_out = pipeline.createXLinkOut()
        left_out.setStreamName("left_out")
        #left.out.link(left_out.input)
        left.out.link(depth.left)  
        right_out = pipeline.createXLinkOut()
        right_out.setStreamName("right_out")
        #right.out.link(right_out.input)
        right.out.link(depth.right)
        depth_out = pipeline.createXLinkOut()
        depth_out.setStreamName("depth_out")
        #depth.disparity.link(depth_out.input)  

        depth.rectifiedLeft.link(left_out.input)
        depth.rectifiedRight.link(right_out.input)
        depth.depth.link(depth_out.input)

        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(str(Path(self.pd_path).resolve().absolute()))
        pd_in = pipeline.createXLinkIn()
        pd_in.setStreamName("pd_in")
        pd_in.out.link(pd_nn.input)
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)

        print("Creating Hand Landmark Neural Network...")          
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(str(Path(self.lm_path).resolve().absolute()))
        self.lm_input_length = 224
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)

        spatial_calculator = pipeline.createSpatialLocationCalculator()
        spatial_calculator.setWaitForConfigInput(True)
        spatial_calculator.inputDepth.setBlocking(False)
        spatial_calculator.inputDepth.setQueueSize(1)

        spatial_data_out = pipeline.createXLinkOut()
        spatial_data_out.setStreamName("spatial_data_out")
        spatial_data_out.input.setQueueSize(1)
        spatial_data_out.input.setBlocking(False)

        spatial_calc_config_in = pipeline.createXLinkIn()
        spatial_calc_config_in.setStreamName("spatial_calc_config_in")

        depth.depth.link(spatial_calculator.inputDepth)
        #spatial_calculator.passthroughDepth.link(depth_out.input)

        spatial_calculator.out.link(spatial_data_out.input)
        spatial_calc_config_in.out.link(spatial_calculator.inputConfig)

        print("Pipeline created.")
        return pipeline

    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,18)) # 896x18
        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors)
        # Non maximum suppression
        self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        mpu.detections_to_rect(self.regions)
        mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)

    def lm_postprocess(self, region, inference):
        region.lm_score = inference.getLayerFp16("Identity_1")[0]    
        region.handedness = inference.getLayerFp16("Identity_2")[0]
        lm_raw = np.array(inference.getLayerFp16("Squeeze"))
        
        lm = []
        for i in range(int(len(lm_raw)/3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3*i:3*(i+1)]/self.lm_input_length)
        region.landmarks = lm

    def lm_render(self, frame, original_frame, region):
        cropped_frame = None
        hand_bbox = []
        index_point = []
        if region.lm_score > self.lm_score_threshold:
            palmar = True
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)

            index_point = [lm_xy[8][0], lm_xy[8][1]]
            if self.show_finger_index:
                pt = lm_xy[8]
                cv2.circle(frame, (pt[0], pt[1]), 3, (255, 0, 255), -1)

            # Calculate the bounding box for the entire hand
            max_x = 0
            max_y = 0
            min_x = frame.shape[1]
            min_y = frame.shape[0]
            for x,y in lm_xy:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

            box_width = max_x - min_x
            box_height = max_y - min_y
            x_center = min_x + box_width / 2
            y_center = min_y + box_height / 2

            # Enlarge the hand bounding box for drawing use
            draw_width = box_width/2 * 1.2
            draw_height = box_height/2 * 1.2
            draw_size = max(draw_width, draw_height)

            draw_min_x = int(x_center - draw_size)
            draw_min_y = int(y_center - draw_size)
            draw_max_x = int(x_center + draw_size)
            draw_max_y = int(y_center + draw_size)

            hand_bbox = [draw_min_x, draw_min_y, draw_max_x, draw_max_y]

            if self.show_hand_box:
                cv2.rectangle(frame, (draw_min_x, draw_min_y), (draw_max_x, draw_max_y), (36, 152, 0), 2)

        return cropped_frame, region.handedness, hand_bbox, index_point

    def query_spatial(self, rects):
        conf_datas = []
        for r in rects:
            conf_data = dai.SpatialLocationCalculatorConfigData()
            conf_data.depthThresholds.lowerThreshold = 100
            conf_data.depthThresholds.upperThreshold = 10000
            conf_data.roi = r
            conf_datas.append(conf_data)
        if len(conf_datas) > 0:
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.setROIs(conf_datas)
            
            self.q_spatial_config.send(cfg)

            # Receives spatial locations
            spatial_data = self.q_spatial_data.get().getSpatialLocations()
            #for i,sd in enumerate(spatial_data):
            #print(f'{i} : {sd.spatialCoordinates.x},{sd.spatialCoordinates.y},{sd.spatialCoordinates.z}')
            '''self.hands[i].xyz_zone =  [
                int(sd.config.roi.topLeft().x) - self.crop_w,
                int(sd.config.roi.topLeft().y),
                int(sd.config.roi.bottomRight().x) - self.crop_w,
                int(sd.config.roi.bottomRight().y)
                ]
            self.hands[i].xyz = [
                sd.spatialCoordinates.x,
                sd.spatialCoordinates.y,
                sd.spatialCoordinates.z
                ]'''
            return spatial_data
        return []

    # calculate interpolated depth of ground from 4 marker points
    def get_interpolated_playground(self,x,y):
        points = np.array([[0,0],[0,1],[1,1],[1,0]])
        values = np.array([self.markerCoordinates[0].z,self.markerCoordinates[1].z,self.markerCoordinates[2].z,self.markerCoordinates[3].z])
        result = griddata(points, values, ([x,y]),  method='linear')[0]
        return result

    def get_playground_affine_transform(self):
        srcTri = np.array( [[self.markerCoordinates[0].x, self.markerCoordinates[0].y], [self.markerCoordinates[1].x, self.markerCoordinates[1].y], [self.markerCoordinates[2].x, self.markerCoordinates[2].y]] ).astype(np.float32)
        srcTri = srcTri * np.array([self.preview_width,self.preview_height]).astype(np.float32) # denormalize
        dstTri = np.array( [[0., 0.], [0.,self.playground_height], [self.playground_width,self.playground_height]] ).astype(np.float32)
        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        return warp_mat

    def get_playground_perspective_transform(self):
        srcQuad = np.array( [[self.markerCoordinates[0].x, self.markerCoordinates[0].y], [self.markerCoordinates[1].x, self.markerCoordinates[1].y], [self.markerCoordinates[2].x, self.markerCoordinates[2].y], [self.markerCoordinates[3].x, self.markerCoordinates[3].y]] ).astype(np.float32)
        srcQuad = srcQuad * np.array([self.preview_width,self.preview_height]).astype(np.float32) # denormalize
        dstQuad = np.array( [[0., 0.], [0.,self.playground_height], [self.playground_width,self.playground_height], [self.playground_width, 0.]] ).astype(np.float32)
        pers_mat = cv2.getPerspectiveTransform(srcQuad, dstQuad)
        return pers_mat

    def run(self):
        device = dai.Device(self.create_pipeline())

        q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        l_video = device.getOutputQueue(name="left_out", maxSize=1, blocking=False)
        r_video = device.getOutputQueue(name="right_out", maxSize=1, blocking=False)
        x_video = device.getOutputQueue(name="depth_out", maxSize=1, blocking=False)

        q_pd_in = device.getInputQueue(name="pd_in")
        q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
        q_lm_in = device.getInputQueue(name="lm_in")
        q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)

        self.q_spatial_data = device.getOutputQueue(name="spatial_data_out", maxSize=4, blocking=False)
        self.q_spatial_config = device.getInputQueue("spatial_calc_config_in")

        # load playground image
        self.playground_img = cv2.imread('playground.png')

        while True:

            video_frame = q_video.get().getCvFrame()
            left_frame = l_video.get().getCvFrame()
            right_frame = r_video.get().getCvFrame()
            depth_frame = x_video.get().getCvFrame()
            depth_frame_Color = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_frame_Color = cv2.equalizeHist(depth_frame_Color)
            depth_frame_Color = cv2.applyColorMap(depth_frame_Color, cv2.COLORMAP_HOT)

            lr_frame = cv2.hconcat([left_frame, right_frame])

            h, w = video_frame.shape[:2]
            self.frame_size = max(h, w)
            self.pad_h = int((self.frame_size - h)/2)
            self.pad_w = int((self.frame_size - w)/2)

            video_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)

            frame_nn = dai.ImgFrame()
            frame_nn.setWidth(self.pd_input_length)
            frame_nn.setHeight(self.pd_input_length)
            frame_nn.setData(to_planar(video_frame, (self.pd_input_length, self.pd_input_length)))
            q_pd_in.send(frame_nn)

            annotated_frame = video_frame.copy()
            
            # Get palm detection
            inference = q_pd_out.get()
            self.pd_postprocess(inference)

            # Send data for hand landmarks
            for i,r in enumerate(self.regions):
                img_hand = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_input_length, self.lm_input_length)
                nn_data = dai.NNData()   
                nn_data.setLayer("input_1", to_planar(img_hand, (self.lm_input_length, self.lm_input_length)))
                q_lm_in.send(nn_data)

            # Retrieve hand landmarks
            index_roi_list = []
            for i,r in enumerate(self.regions):
                inference = q_lm_out.get()
                self.lm_postprocess(r, inference)
                hand_frame, handedness, hand_bbox, index_point = self.lm_render(video_frame, annotated_frame, r)
                
                # preprocess index finger points for spatial calc.
                if len(index_point) > 0:
                    # un-padded location
                    x0 = index_point[0] - self.pad_w
                    y0 = index_point[1] - self.pad_h
                    # make roi
                    topLeft = dai.Point2f(((x0-self.roi_size)/self.preview_width), ((y0-self.roi_size)/self.preview_height))
                    bottomRight = dai.Point2f(((x0+self.roi_size)/self.preview_width), ((y0+self.roi_size)/self.preview_height))
                    # append to roi list
                    index_roi_list.append(dai.Rect(topLeft, bottomRight))

            # Remove padding
            video_frame = video_frame[self.pad_h:self.pad_h+h, self.pad_w:self.pad_w+w]

            marker_roi_list = []
            (mk_corners, mk_ids, mk_rejected) = cv2.aruco.detectMarkers(video_frame, self.arucoDict, parameters=self.arucoParams)
            cv2.aruco.drawDetectedMarkers(video_frame, mk_corners, mk_ids)
            for i,c in enumerate(mk_corners):
                # preprocess marker points for spatial calc.
                x0 = int(c[0][0][0])
                y0 = int(c[0][0][1])
                # make roi
                topLeft = dai.Point2f(((x0-self.roi_size)/self.preview_width), ((y0-self.roi_size)/self.preview_height))
                bottomRight = dai.Point2f(((x0+self.roi_size)/self.preview_width), ((y0+self.roi_size)/self.preview_height))
                # append to roi list
                marker_roi_list.append(dai.Rect(topLeft, bottomRight))

                # Draw  markers
                cv2.circle(depth_frame_Color, (x0, y0), 3, (255, 255, 0), -1)
                cv2.putText(depth_frame_Color,f"{int(mk_ids[i])}",(x0 - 15, y0 - 15),1,1,(255, 255, 0),1)
                #cv2.rectangle(depth_frame_Color, (x1,y1),(x2,y2), (255, 0, 0), 2)

            # Append ROI list for calculation
            index_roi_list = marker_roi_list + index_roi_list

            # Calculate spatial data from ROI list
            spatial_data = self.query_spatial(index_roi_list)

            # Post process spatial data
            marker_found = len(mk_corners)
            self.fingerCoordinates = [] # force update fingers data
            for i,sd in enumerate(spatial_data):
                roi = sd.config.roi#.denormalize(self.preview_width,self.preview_height)
                if i < marker_found: # update marker data if found
                    ids_to_idx = int(mk_ids[i]-1)
                    if (ids_to_idx >= 0) and (ids_to_idx < 4) :
                        self.markerCoordinates[ids_to_idx].x = roi.x + (roi.width/2) # normalized screen space
                        self.markerCoordinates[ids_to_idx].y = roi.y + (roi.height/2) # normalized screen space
                        self.markerCoordinates[ids_to_idx].z = sd.spatialCoordinates.z # spatial space
                        #print(f"M {i} : {self.markerCoordinates[ids_to_idx].x},{self.markerCoordinates[ids_to_idx].y},{self.markerCoordinates[ids_to_idx].z}")
                else: # everything else is fingers
                    fingerCoord = dai.Point3f()
                    fingerCoord.x = roi.x + (roi.width/2) # normalized screen space
                    fingerCoord.y = roi.y + (roi.height/2) # normalized screen space
                    fingerCoord.z = sd.spatialCoordinates.z # spatial space
                    self.fingerCoordinates.append(fingerCoord)
                    #print(f"F {i} : {fingerCoord.x},{fingerCoord.y},{fingerCoord.z}")

            # Draw all spatial ROI data on depth map
            for i,sd in enumerate(spatial_data):
                roi = sd.config.roi.denormalize(self.preview_width,self.preview_height)
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)
                #print(f'{x1},{y1} = {sd.spatialCoordinates.z}')
                cv2.rectangle(depth_frame_Color, (x1,y1),(x2,y2), (255, 0, 0), 2)
                cv2.putText(depth_frame_Color,f"Z: {int(sd.spatialCoordinates.z)} mm",(x1 + 10, y1 + 10),1,1,(0, 255, 0),1)

            # calculate warp playground image
            aff_mat = self.get_playground_affine_transform()
            warp_frame = cv2.warpAffine(video_frame, aff_mat, (self.preview_width,self.preview_height))

            # calculate simple touches
            per_mat = self.get_playground_perspective_transform()
            for i,fingerCoord in enumerate(self.fingerCoordinates):
                ground_under = self.get_interpolated_playground(fingerCoord.x,fingerCoord.y)
                if fingerCoord.z >= ground_under - self.touch_treshold :
                    #print(f"{ground_under} : {fingerCoord.z}")
                    # calc transform touch point
                    old_coord = np.array([[[fingerCoord.x*self.preview_width, fingerCoord.y*self.preview_height]]]) # denormalize
                    touch_point = cv2.perspectiveTransform(old_coord, per_mat)[0][0]
                    print(f"touch at : {touch_point}")
                    cv2.circle(self.playground_img, (int(touch_point[0]),int(touch_point[1])), 20, (255, 255, 0), -1)
            
            #cv2.imshow("video_frame", video_frame)
            #cv2.imshow("depth_frame", depth_frame_Color)
            #cv2.imshow("warp_frame", warp_frame)
            #cv2.imshow("lr_frame", lr_frame)
            t = cv2.hconcat([video_frame,depth_frame_Color])
            b = cv2.hconcat([warp_frame,self.playground_img])
            debug_img = cv2.vconcat([t,b])
            cv2.imshow("debug", debug_img)

            cv2.namedWindow("playground", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("playground", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
            cv2.imshow("playground", self.playground_img)

            key = cv2.waitKey(1) 
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord('r'):
                # reload playground
                self.playground_img = cv2.imread('playground.png')
                pass

if __name__ == "__main__":
    mk = Marker()
    mk.run()