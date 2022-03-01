import os
import cv2
import time

def Write_Detection_Result(detect_result, lm_path):
    '''
    Usage:
        Write the detection result from MTCNN to .txt format
    Args:
        detect_result: (dict) of the detection results
        lm_path: (str) of the file path with .txt as the extension
    '''
    
    mfile = open(lm_path, 'w')
    for val in detect_result[0]['keypoints'].values():
        str_to_write = str(val[0]) + ' ' + str(val[1]) + '\n'
        mfile.write(str_to_write)


def Extract_Landmark(detector, discofacegan_save_dir, ffhq_image_dir):
    preprocessed_lm_dir = os.path.join(discofacegan_save_dir, 'LandMark')
    if not os.path.exists(preprocessed_lm_dir):
        os.mkdir(preprocessed_lm_dir)

    start_time = time.time()
    for img_file in os.listdir(ffhq_image_dir):
        try:
            img_path = os.path.join(ffhq_image_dir, img_file) 
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            detect_result = detector.detect_faces(img)
            lm_path = os.path.join(preprocessed_lm_dir, img_file.replace('png', 'txt'))

            Write_Detection_Result(detect_result, lm_path)
        except:
            print('Oops, image file: ' + str(img_file) + ' has processing issues.')

    end_time = time.time()
    print('Land mark extraction process time: ' + str(round(end_time - start_time, 2)))    
    
    return preprocessed_lm_dir
