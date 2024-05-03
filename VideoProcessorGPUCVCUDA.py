import cvcuda
import tensorrt as trt
from datetime import datetime
import yaml
import numpy as np
import torch
from torch2trt import TRTModule
import cv2
class VideoProcessor:
    def __init__(self) -> None: 
        self.config = yaml.load(open("../gpupipe/config/data.yaml"), Loader=yaml.FullLoader)
        self.modelName = self.config['modelName']
        self.modelVersion = self.config['modelVersion']
        self.inputName = self.config['inputName']
        self.outputName = self.config['outputName']
        self.confidenceThres = self.config['confidenceThreshold']
        self.inputWidth, self.inputHeight = self.config['inputWidth'],self.config['inputHeight']
        self.iouThres = self.config['iouThreshold']
        self.classes = self.config["names"]
        self.colorPalette = np.random.uniform(0, 255, size=(len(self.classes), 3)).astype(np.uint8)
        # create a FPS counter
        self.fps = 0
        self.fpsCounter = 0
        self.fpsTimer = datetime.now()
        # Initalize TensorRT Engine
        self.logger = trt.Logger(trt.Logger.INFO)
        with open("../gpupipe/model/best.engine","rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.TRTNet = TRTModule(input_names=[self.inputName],output_names=[self.outputName],engine=self.engine)
    def preprocess(self,imageFrame):
        # convet the image to a cuda tensor
        imageFrame = torch.tensor(imageFrame,device="cuda",dtype=torch.uint8)
        self.imageHeight,self.imageWidth = imageFrame.shape[:2]
        imageTensor = cvcuda.as_tensor(imageFrame,"HWC")
        imageTensor = cvcuda.cvtcolor(imageTensor,cvcuda.ColorConversion.BGR2RGB)
        imageTensor = cvcuda.resize(imageTensor,(self.inputWidth,self.inputHeight,3))
        # convert torch tensor to numpy array
        imageData = torch.as_tensor(imageTensor.cuda(),device="cuda")
        imageData = imageData / 255.0
        imageData = imageData.transpose(0,2).transpose(1,2).cpu().numpy()
        imageData = np.expand_dims(imageData,axis=0).astype(np.float32)
        return imageData
    def CUDARender(
        self,
        inputFrame:np.ndarray,
        nms_masks:torch.Tensor,
        bbox:torch.Tensor,
        classIndex:torch.Tensor,
        score:torch.Tensor
    ):
        if isinstance(inputFrame,np.ndarray):
            frame_hwc = cvcuda.as_tensor(
                torch.as_tensor(inputFrame).cuda(),
                "HWC"
            )
        bounding_boxes_list = []
        label_text_list = []
        for current_boxes,current_masks,current_class_id,current_score in zip(bbox,nms_masks,classIndex,score):
            filtered_boxes = current_boxes[current_masks]
            # Save the count of non-zero bounding boxes of the image
            bounding_boxes = []
            label_text = []

            for box in filtered_boxes:
                bounding_boxes.append(
                    cvcuda.BndBoxI(
                        box = tuple(box),
                        thickness = 2,
                        borderColor = tuple(self.colorPalette[current_class_id]),
                        fillColor = (0,0,0,0)
                    )
                )
                label_text.append(
                    cvcuda.Label(
                        utf8Text = '{}: {}'.format(self.classes[current_class_id],str(float(current_score.amax().cpu().numpy()) * 100)[0:5] + '%'),
                        fontSize = 1,
                        tlPos = (int(box[0]),int(box[1])),
                        fontColor = (255,255,255),
                        bgColor = tuple(self.colorPalette[current_class_id].tolist())
                    )
                )
            bounding_boxes_list.append(bounding_boxes)
            label_text_list.append(label_text)
        
        batch_bounding_boxes = cvcuda.Elements(elements=bounding_boxes_list)
        batch_label_text = cvcuda.Elements(elements=label_text_list)

        cvcuda.osd_into(frame_hwc,frame_hwc,batch_bounding_boxes)
        cvcuda.osd_into(frame_hwc,frame_hwc,batch_label_text)

        # Copy frame_hwc to ndarray
        inputFrame = torch.as_tensor(frame_hwc.cuda()).cpu().numpy()
        return inputFrame

    def postProcess(self,inputFrame,output):
        output = torch.transpose(torch.squeeze(output),0,1).cuda()
        x_factor = self.imageWidth / self.inputWidth
        y_factor = self.imageHeight / self.inputHeight

        # Process model output
        argmax = torch.argmax(output[:,4:84],dim=1)
        amax = torch.max(output[:,4:84],dim=1).values

        # Concate tensors
        output = torch.cat((output,torch.unsqueeze(argmax,1),torch.unsqueeze(amax,1)),dim=1)
        output = output[output[:,-1] > self.confidenceThres]

        boxes = output[:,:4]
        class_ids = output[:,-2]
        scores = output[:,-1]

        boxes[:,0] = (boxes[:,0] - boxes[:,2]/2.0) * x_factor
        boxes[:,1] = (boxes[:,1] - boxes[:,3]/2.0) * y_factor
        boxes[:,2] = boxes[:,2] * x_factor
        boxes[:,3] = boxes[:,3] * y_factor

        # Convert to boxes dtype to 16bit Signed Integer
        boxes = boxes.to(torch.int16).reshape(1,-1,4)
        scores = scores.to(torch.float32).reshape(1,-1)
        class_ids = class_ids.to(torch.int16)
        # Converting to cvcuda tensor
        cvcuda_boxes = cvcuda.as_tensor(boxes.contiguous().cuda())
        cvcuda_scores = cvcuda.as_tensor(scores.contiguous().cuda())
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        nms_masks = cvcuda.nms(cvcuda_boxes,cvcuda_scores,self.confidenceThres,self.iouThres)
        nms_masks_pyt = torch.as_tensor(
            nms_masks.cuda(),device="cuda",dtype=torch.bool
        )

        # Render the bounding boxes
        outputFrame = self.CUDARender(
            inputFrame,
            nms_masks_pyt,
            boxes,
            class_ids,
            scores
        )

        # calculate the FPS
        self.fpsCounter += 1
        elapsed = (datetime.now() - self.fpsTimer).total_seconds()
        if elapsed > 1.0:
            self.fps = self.fpsCounter / elapsed
            self.fpsCounter = 0
            self.fpsTimer = datetime.now()
        # draw the FPS counter
        cv2.putText(outputFrame, "FPS: {:.2f}".format(self.fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1, cv2.LINE_AA)
        # draw current time on the top right of frame
        cv2.putText(outputFrame, datetime.now().strftime("%Y %I:%M:%S%p"), (self.imageWidth - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1, cv2.LINE_AA)
        return outputFrame
    def inference(self,frame):
        frame = torch.from_numpy(frame).cuda()
        return self.TRTNet(frame)[0]
    def processing(self,frame):
        image_data = self.preprocess(frame)
        output = self.inference(image_data)
        outputFrame = self.postProcess(frame,output)
        return outputFrame
    