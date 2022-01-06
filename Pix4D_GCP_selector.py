import json
import cv2
import numpy as np
import computer_vision_utils as cv_util
import math

def load_image(path):
    image = cv2.imread(path)
    return image

def get_pixel_location(event,x,y,flags,param):
    global mouse_locs,circles

    if event == cv2.EVENT_MBUTTONDOWN:        
        circles.append((x,y))

    if event == cv2.EVENT_LBUTTONDBLCLK:
        
        lonlat = input("Enter lon and lat with comma in between: ")
        lon = float(lonlat.split(",")[0])
        lat = float(lonlat.split(",")[1])
        mouse_locs.append({"x":x,"y":y,"lon":lon,"lat":lat})

        print('>>> point captured at ({0},{1}) for lid {2}.'.format(x,y,lonlat))

def visualize_and_select_coords(img):
    cv2.namedWindow("win",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("win",1600,800)
    cv2.setMouseCallback('win',get_pixel_location)
    
    while True:
        for c in circles:
            cv2.circle(img,c,5,(255,0,0),-1)
            
        cv2.imshow("win",img)
        res = cv2.waitKey(100)
        if res == ord('q'):
            break

def calculate_RMSE(selected_coords):
    src = []
    dst = []

    for coord in selected_coords:
        src.append([coord['x'],coord['y']])
        dst.append([coord['lon'],coord['lat']])

    H,_ = cv2.findHomography(np.array(src), np.array(dst), maxIters = 1000, confidence = 0.8, method=0)

    print(H)
    GPS_diffs = []

    for coord in selected_coords:

        p = np.array([coord['x'],coord['y'],1])
        p_new = np.matmul(H,p)
        p_new/=p_new[2]

        GT = [coord['lon'],coord['lat']]
        
        GPS_diffs.append(cv_util.get_gps_distance(GT[1],GT[0],p_new[1],p_new[0])**2)

    print(np.sqrt(GPS_diffs))
    return math.sqrt(np.mean(GPS_diffs))

def save_coords(coords,path):
    with open(path,'w') as f:
        json.dump(coords,f)

# mouse_locs = [{'x':107,'y':1609,'lon':-111.97497384904608,'lat':33.074630234935185},{'x':172,'y':1610,'lon':-111.9748645746704,'lat':33.074630277554704},{'x':176,'y':1488,'lon':-111.97487527775284,'lat':33.07477505086456},{'x':110,'y':1487,'lon':-111.97497373287044,'lat':33.074775092832915},{'x':116,'y':1124,'lon':-111.97497388815981,'lat':33.07520677625109},{'x':119,'y':1002,'lon':-111.97497355079616,'lat':33.075351004483764},{'x':185,'y':1004,'lon':-111.97487530828975,'lat':33.07535087066245},{'x':122,'y':760,'lon':-111.97497437283774,'lat':33.0756391786435},{'x':188,'y':761,'lon':-111.97487506075876,'lat':33.075639233583544},{'x':190,'y':640,'lon':-111.9748754691418,'lat':33.075783095187816},{'x':124,'y':639,'lon':-111.9749736040094,'lat':33.075783139648145},{'x':126,'y':396,'lon':-111.97497346395953,'lat':33.076071501706146},{'x':127,'y':153,'lon':-111.974974501314,'lat':33.076359764347146},{'x':128,'y':93,'lon':-111.9749739036563,'lat':33.076431836706625},{'x':193,'y':94,'lon':-111.97487576806832,'lat':33.07643179225987}]
mouse_locs = []
circles = []
path = "/home/ariyan/Desktop/GRG.png"
coord_path = "/home/ariyan/Desktop/GRG.json"
img = load_image(path)
visualize_and_select_coords(img)
save_coords(mouse_locs,coord_path)
RMSE = calculate_RMSE(mouse_locs)
print(RMSE)