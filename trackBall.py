
#OpenCV tracking ball
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def isOpen(f):
    return plt.fignum_exists(f.number)
def calc_center_pixels():
    """
    Input: none
    This function applies a mask to the image and calculates the center of 
    the ball in pixel coordinates.
    Output: np.array(lxmidp),np.array(lymidp),np.array(rxmidp),np.array(rymidp)
    """
    # load in video
    leftv = cv.VideoCapture('left.mp4')
    Nframes = int(leftv.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(leftv.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(leftv.get(cv.CAP_PROP_FRAME_HEIGHT))

    rightv = cv.VideoCapture('right.mp4')
    Nframes2 = int(rightv.get(cv.CAP_PROP_FRAME_COUNT))

    # set up plotting
    plt.ion()
    fig, ax = plt.subplots(2,2)
    lh_img = ax[0][0].imshow(np.zeros((height,width,3)))
    h_ball = ax[1][0].imshow(np.ones((height,width)), cmap='gray', vmin=0, vmax=1)
    h_center = ax[1][0].plot(0,0,'*r')[0]
    rh_img = ax[0][1].imshow(np.zeros((height,width,3)))
    h_ball2 = ax[1][1].imshow(np.ones((height,width)), cmap='gray', vmin=0, vmax=1)
    h_center2 = ax[1][1].plot(0,0,'*r')[0]
    plt.show()

    # start processing video
    lxmidp=[]
    

    lymidp=[]
    
    rxmidp=[]
    

    rymidp=[]
    

    while leftv.isOpened() and isOpen(fig) and rightv.isOpened() :
        has_frame, limg = leftv.read()
        has_frame2, rimg = rightv.read()



        if has_frame and has_frame2:
            nframe = int(leftv.get(cv.CAP_PROP_POS_FRAMES))
            n2frame = int(rightv.get(cv.CAP_PROP_POS_FRAMES))
            # display video
            limg_plt = cv.cvtColor(limg, cv.COLOR_BGR2RGB)
            rimg_plt = cv.cvtColor(rimg, cv.COLOR_BGR2RGB)


            lh_img.set_data(limg_plt)
            rh_img.set_data(rimg_plt)


            ax[0][0].set_title(f'frame {str(nframe)} / {str(Nframes)}\nleft camera')
            ax[0][1].set_title(f'frame {str(n2frame)} / {str(Nframes2)}\nright camera')
            
            ax[1][0].set_title('Left Mask')
            ax[1][1].set_title('Right Mask')

            # display mask
            limg_hsv = cv.cvtColor(limg, cv.COLOR_BGR2HSV)
            rimg_hsv = cv.cvtColor(rimg, cv.COLOR_BGR2HSV)

            

            mask_g = (limg_hsv[:,:,0] > 3) & (limg_hsv[:,:,0] < 30)
            mask_s = limg_hsv[:,:,1] > 150
            mask_v = limg_hsv[:,:,2] > 150
            mask = mask_g & mask_s & mask_v
            h_ball.set_data(1*mask)

            mask_g2 = (rimg_hsv[:,:,0] > 0) & (rimg_hsv[:,:,0] < 30)
            mask_s2 = rimg_hsv[:,:,1] > 150
            mask_v2 = rimg_hsv[:,:,2] > 150
            mask2 = mask_g2 & mask_s2 & mask_v2
            h_ball2.set_data(1*mask2)

            # some computation - what does it do?
            row,col = np.where(mask)
            x_mid = np.mean(col)
            lxmidp.append(x_mid)
            y_mid = np.mean(row)
            lymidp.append(y_mid)
            h_center.set_data(x_mid,y_mid)

            row2,col2 = np.where(mask2)
            x_mid2 = np.mean(col2)
            rxmidp.append(x_mid2)
            y_mid2 = np.mean(row2)
            rymidp.append(y_mid2)
            h_center2.set_data(x_mid2,y_mid2)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            break

    leftv.release()
    rightv.release()
    cv.destroyAllWindows()
    
    return np.array(lxmidp),np.array(lymidp),np.array(rxmidp),np.array(rymidp)



def calc_center_inches():
    """
    Input: none
    This function calculates the center of the ball in image coordinates (inches)
    Output: time,lxmidi,rxmidi,lymidi,rymidi
    """
    lxmidp,lymidp,rxmidp,rymidp = calc_center_pixels()
    leftv = cv.VideoCapture('left.mp4')
    width = int(leftv.get(cv.CAP_PROP_FRAME_WIDTH)) # width to calculating shfting
    Nframes = int(leftv.get(cv.CAP_PROP_FRAME_COUNT))  # Frame of video
    fps = int(leftv.get(cv.CAP_PROP_FPS)) # fps of video
    time = Nframes/fps # real count of time 
    height=int(leftv.get(cv.CAP_PROP_FRAME_HEIGHT))  # height to calculating shfting
    scale =0.02 # in/pixel

    lxmidi=[]
    rxmidi=[]

    lymidi=[]
    rymidi=[]

    #calculating x centre coord in inches
    for c in range (len(lxmidp)):
        tempxi = (lxmidp[c]-(width/2))*scale
        lxmidi.append(tempxi)
    
    for c in range (len(rxmidp)):
        tempxi = (rxmidp[c]-(width/2))*scale
        rxmidi.append(tempxi)
    #calculating y centre coord in inches
    for c in range(len(lymidp)):
        tempyi= -(lymidp[c]-(height/2))*scale
        lymidi.append(tempyi)
    
    for c in range(len(rymidp)):
        tempyi= -(rymidp[c]-(height/2))*scale
        rymidi.append(tempyi)
        
    return time,lxmidi,rxmidi,lymidi,rymidi
    
    

def calc_3dtrajectory():
    """
    Input: none
    This function calculates the x, y, z 
    positions of the ball in scene coordinates (inches) over time.
    Output: time,lxmidi,rxmidi,lymidi,rymidi,xinch,yinch,z
    """
    f = 3 #inches
    d = 3 #inches
    time,lxmidi,rxmidi,lymidi,rymidi=calc_center_inches()

    #calculating z in real world coords
    z = []
    for c in range (len(lxmidi)):
        tempz = (f*d)/((lxmidi[c])-(rxmidi[c]))
        z.append(tempz)
    #calculating x in real world coords
    xinch = []
    for c in range (len(z)):
        tempx = (z[c]/f)*(lxmidi[c]) - (d/2)
        xinch.append(tempx)
    np.array(xinch)

    #calculating y in real world coords
    yinch = []
    for c in range (len(z)):
        tempy = (z[c]/f)*(lymidi[c])
        yinch.append(tempy)
    return time,lxmidi,rxmidi,lymidi,rymidi,xinch,yinch,z

def plot():
    """
    Input: none
    plots our values
    Output: none
    """
    time,lxmidi,rxmidi,lymidi,rymidi,xinch,yinch,z=calc_3dtrajectory()
    
    plt.ioff() # turning off interactive plots

    # Create two subplots side by side
    fig2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the data on the first subplot
    ax1.plot(lxmidi,lymidi)
    ax1.set_xlabel("X position (in)")
    ax1.set_ylabel("Y position (in)")
    ax1.set_title('Left Camera Ball center')

    # Plot the data on the second subplot
    ax2.plot(rxmidi,rymidi)
    ax2.set_xlabel("X position (in)")
    ax2.set_ylabel("Y position (in)")
    ax2.set_title('Right Camera Ball center')

    # Show the plots
    plt.show()
    while isOpen(fig2d)==True:
        plt.pause(1)
    # plots x,y,z over time
    txyz =plt.figure()
    plt.plot(np.linspace(0,time,len(xinch)),xinch,label="x")
    plt.plot(np.linspace(0,time,len(xinch)),yinch,label="y")
    plt.plot(np.linspace(0,time,len(xinch)),z,label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (in)")
    plt.title("2D plot of Ball Trajectory")
    plt.legend()
    plt.show()
    while isOpen(txyz)==True:
        plt.pause(1)
    #plots x,y,z as 3d plot
    fig3 =plt.figure()
    ax=fig3.add_subplot(111,projection='3d')
    ax.plot(xinch,yinch,z)
    ax.set_title("3D plot of Ball Trajectory")
    ax.set_xlabel("X position (in)")
    ax.set_ylabel("Y position (in)")
    ax.set_zlabel("Z  position (in)")
    while isOpen(fig3)==True:
        plt.pause(1)
    
plot()












