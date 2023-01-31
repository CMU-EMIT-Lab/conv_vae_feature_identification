import cv2
import matplotlib.pyplot as plt
import skimage
from map_model import *


def non_max_suppression(boxes, overlap_thresh=.5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1, yy1, xx2, yy2 = (
                                 np.maximum(x1[i], x1[idxs[:last]]),
                                 np.maximum(y1[i], y1[idxs[:last]]),
                                 np.minimum(x2[i], x2[idxs[:last]]),
                                 np.minimum(y2[i], y2[idxs[:last]])
        )
        # compute the width and height of the bounding box
        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


def get_img_prediction_bounding_box(path, model, params):
    img = get_array(path, params)
    img = img.reshape(1, img.shape[0], img.shape[1], 1)
    pred = model.predict(img)[0]
    category_dict = {0: 'Nothing', 1: 'Useful Feature'}
    cat_index = np.argmax(pred)
    cat = category_dict[cat_index]
    print(f'{path}\t\tPrediction: {cat}\t{int(pred.max()*100)}% Confident')

    #speed up cv2
    cv2.setUseOptimized(True)
    cv2.setNumThreads(10) # change depending on your computer
    img = cv2.imread(path)
    clone = img.copy()
    clone2 = img.copy()
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.addStrategy(cv2.ximgproc.segmentation.SelectiveSearchSegmentationStrategyColor())
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    windows = []
    locations = []
    print(f'Creating Bounding Boxes for {path}')
    for x, y, w, h in rects[:11]:
        startx, starty, endx, endy = x, y, x+w, y+h
        roi = img[starty:endy, startx:endx]
        roi = cv2.resize(roi, dsize=(params.image_size, params.image_size), interpolation=cv2.INTER_CUBIC)
        windows.append(roi)
        locations.append((startx, starty, endx, endy))
    windows = np.array(windows)
    windows = windows.reshape(int(windows.shape[0]*3), windows.shape[1], windows.shape[2], 1)
    windows = np.array(windows)
    locations = np.array(locations)
    preds = model.predict(windows)
    nms = non_max_suppression(locations)
    bounding_cnt = 0
    for idx in nms:
        if np.argmax(predictions[idx]) != cat_index:
            continue
        startx, starty, endx, endy = locations[idx]
        cv2.rectangle(clone, (startx, starty), (endx, endy), (0,0,255), 2)
        text = f'{category_dict[np.argmax(preds[idx])]}: {int(preds[idx].max()*100)}%'
        cv2.putText(clone, text, (startx, starty+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
        bounding_cnt += 1

    if bounding_cnt == 0:
        pred_idx= [idx for idx, i in enumerate(preds) if np.argmax(i) == cat_index]
        cat_locations = np.array([locations[i] for i in pred_idx])
        nms = non_max_suppression(cat_locations)
        if len(nms)==0:
            cat_predictions = preds[:, cat_index]
            pred_max_idx = np.argmax(cat_predictions)
            pred_max = cat_predictions[pred_max_idx]
            pred_max_window = locations[pred_max_idx]
            startx, starty, endx, endy = pred_max_window
            cv2.rectangle(clone, (startx, starty), (endx, endy),  (0,0,255),2)
            text = f'{category_dict[cat_index]}: {int(pred_max*100)}%'
            cv2.putText(clone, text, (startx, starty+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
        for idx in nms:
            startx, starty, endx, endy = cat_locations[idx]
            cv2.rectangle(clone, (startx, starty), (endx, endy), (0,0,255), 2)
            text = f'{category_dict[np.argmax(predictions[pred_idx[idx]])]}: {int(predictions[pred_idx[idx]].max()*100)}%'
            cv2.putText(clone, text, (startx, starty+15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    cv2.imshow(f'Test', np.hstack([clone, clone2]))
    cv2.waitKey(0)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        ss.clear()
    return predictions


if __name__ == "__main__":
    from train import TrainParams
    check_params = TrainParams(
     parent_dir='data_binary_watermark',
     name='watermark_test',
     epochs=50,
     batch_size=16,
     image_size=128,
     latent_dim=128,
     num_examples_to_generate=16,
     learning_rate=0.001
     # show_latent_gif=True
    )

    normal_model = detection_model(check_params)
    normal_model.load_weights(f'../outputs/{check_params.name}/{check_params.name}_detector_ModelWeights.h5')
    # path to the model weights
    test_folder = f'../mapped/{check_params.name}/full_micrographs'  # folder where you will put your images to test
    predictions = []
    prediction_paths = [
        f'../mapped/{check_params.name}/full_micrographs/{i}' for i in os.listdir(
            f'../mapped/{check_params.name}/full_micrographs'
        ) if not i.startswith('.')
    ]
    for i in prediction_paths:
        prediction = get_img_prediction_bounding_box(i, normal_model, check_params)
        predictions.append(prediction)
