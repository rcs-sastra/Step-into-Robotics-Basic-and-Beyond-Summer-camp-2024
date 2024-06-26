import cv2

def detect_shapes(img):

    detected_shapes = []
    
    # convert to grayscale
    imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    #thresholding
    _, thrash = cv2.threshold(imgGry, 175, 255, cv2.THRESH_BINARY_INV)

    #finding all the contours
    contours , _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    #iterating through contours
    for contour in contours: 
        shape_details=[]
        Shape,Color,cX,cY = None,None,None,None
        
        #detecting shape 
        approx = cv2.approxPolyDP(contour, 0.04* cv2.arcLength(contour, True), True)
        
        if len(approx) == 3 :
            Shape = 'Triangle'
        elif len(approx) == 4 :
            x, y , w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
            if aspectRatio >= 0.95 and aspectRatio < 1.10:
                Shape = 'Square'
            else:
                Shape = 'Rectangle'
        elif len(approx) == 5 :
            Shape = 'Pentagon'
        else:
            Shape = 'Circle'
        
        #detecting mid point
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        
        #detecting the colour in bgr format at mid point of the contour in current iteration...
        bgr=img[cY][cX]
        
        if(bgr[0] in range(240,256) and bgr[1] in range(0,16)  and bgr[2] in range(0,16)):
            Color = 'Blue'
        elif(bgr[1] in range(245,256) and bgr[2] in range(0,16) and bgr[0] in range(0,16)):
            Color = 'Green'
        elif(bgr[2] in range(245,256) and bgr[1] in range(0,16) and bgr[0] in range(0,16)):
            Color = 'Red'
        elif(bgr[1] in range(139,166) and bgr[0] in range(0,16) and bgr[2] in range(250,256)):
            Color=  'Orange'
        elif(bgr[1] in range(250,256) and bgr[0] in range(120,130) and bgr[2] ==0):
            Color = 'Violet'
        elif(bgr[1] == 0 and bgr[0] in range(125,135) and bgr[2] in range(70,80)):
            Color = 'Indigo'
        elif(bgr[1] in range(250,256) and bgr[0] ==0 and bgr[2] ==255):
            Color = 'Yellow'

        shape_details.append(Color)
        shape_details.append(Shape)
        mp = (cX,cY)
        shape_details.append(mp)
        detected_shapes.append(shape_details)

    ##################################################
    
    return detected_shapes

def get_labeled_image(img, detected_shapes):
    for detected in detected_shapes:
        colour = detected[0]
        shape = detected[1]
        coordinates = detected[2]
        cv2.putText(img, str((colour, shape)),coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return img

if __name__ == '__main__':
    
    img_dir_path = 'images_data/'
    for file_num in range(1, 11):
            img_file_path = img_dir_path + 'test_image_' + str(file_num) + '.png'
            img = cv2.imread(img_file_path)
            print('\n****************************************')
            print('\nFor test_image_' + str(file_num) + '.png')
            
            # detect shape properties from image
            detected_shapes = detect_shapes(img)
            print(detected_shapes)
            
            # display image with labeled shapes
            img = get_labeled_image(img, detected_shapes)
            cv2.imshow("labeled_image", img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()




        