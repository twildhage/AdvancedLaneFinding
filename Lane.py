from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import cv2


class Lane():
    
    def __init__(self, nb_memory_items):
        
        self.nb_memory_items = nb_memory_items
        
        self.search_type = 'window'
        # x values of the last n fits of the line

        #x values for detected line pixels
        self.x = {'left': [], 'right': []}  
        #y values for detected line pixels
        self.y = {'left': [], 'right': []} 

        self.recent_xfitted = {'left': deque(maxlen=nb_memory_items), 'right':deque(maxlen=nb_memory_items)}  
        # x values of the current fits of the line
        self.current_xfitted = {'left': [], 'right':[]}
        #average x values of the fitted line over the last n iterations
        self.best_xfitted = {'left': [], 'right': []} 

        
        #polynomial coefficients of last n iterations
        self.recent_fit = {'left': deque(maxlen=nb_memory_items), 'right':deque(maxlen=nb_memory_items)}  
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = {'left': [], 'right':[]}  
        #polynomial coefficients for the most recent fit
        self.current_fit = {'left': [], 'right':[]} 
        
        # Status of the lines tells if they are valid
        self.line_valid = {'left': True, 'right': True} 
        self.invalid_counter = {'left': 0, 'right': 0}
        self.search_margin = {'left': 65, 'right': 65} 
                
        self.ploty  = np.linspace(0, 719, 720) 
        self.ploty2 = self.ploty**2 
        #radius of curvature of the line in some units
        self.radius_of_curvature = {'left': [], 'right':[]} 
        self.average_radius_of_curvature = [] 
        #distance in meters of vehicle center from the line
        self.vehicle_position = 0
        self.xmeter_per_pixel = 3.7/700
        self.ymeter_per_pixel = 30/720
        
        # ++++++++++++++++++++++ N O T  U S E D ++++++++++++++++++++++ 
        # was the line detected in the last iteration?
        self.detected = False  
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # ++++++++++++++++++++++ N O T  U S E D ++++++++++++++++++++++ 

        
    def _analyse_line_status(self, verbose=0):
        if len(self.recent_xfitted['left']) >= 1:
            last_radius = self.radius_of_curvature.copy()
            self._calculate_curvature_radius()
            for key in self.line_valid:
                self.line_valid[key] =  (self.radius_of_curvature[key] > 150) # ((np.absolute(last_radius[key]-self.radius_of_curvature[key]) < 100) &
                if self.line_valid[key] == False:
                    self.invalid_counter[key] += 1
                else:
                    self.invalid_counter[key] = 0

                if self.invalid_counter[key] >= 2:
                    self.search_type = 'window'
                if verbose==1:
                    print("{0}:\t{1},\t valid[{2}],\t margin[{3}],\t counter[{4},\t #pxl[{5}],\tx[{6}]]".format(key,
                                                                                       self.search_type,
                                                                                       self.line_valid[key],
                                                                                       self.search_margin[key],
                                                                                       self.invalid_counter[key],
                                                                                       len(self.x[key]),
                                                                                        self.best_xfitted[key][-1]))
            if verbose == 1:
                print('\n')
                    
                
    def _bounded_increase_search_margin(self, key):
            margin  = self.search_margin[key]
            self.search_margin[key] = min((margin + 50), 200)
            
            
    def _bounded_decrease_search_margin(self, key):
            margin  = self.search_margin[key]
            self.search_margin[key] = max((margin - 25), 65)


        
    def line_search(self, binary_warped, verbose=0):
        
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds  = []
        right_lane_inds = []

        # Step through the windows one by one
        # The search type will be change to 'ancestor' after first window search
        if self.search_type == 'window':
            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[360:720,:], axis=0)

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(binary_warped.shape[0]/nwindows)
            # Current positions to be updated for each window

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50
            
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
             
            # Concatenate the arrays of indices
            left_lane_inds  = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            self.search_type = 'ancestor'
                
        elif self.search_type=='ancestor':
            # Assume you now have a new warped binary image 
            # from the next frame of video (also called "binary_warped")
            # It's now much easier to find line pixels!
            nonzero  = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            A,B,C = self.current_fit['left']
            margin   = self.search_margin['left']
            left_lane_inds = ((nonzerox > (A*(nonzeroy**2) + B*nonzeroy + C - margin)) &
                              (nonzerox < (A*(nonzeroy**2) + B*nonzeroy + C + margin))) 

            A,B,C = self.current_fit['right']
            margin   = self.search_margin['right']
            right_lane_inds = ((nonzerox > (A*(nonzeroy**2) + B*nonzeroy + C - margin)) &
                               (nonzerox < (A*(nonzeroy**2) + B*nonzeroy + C + margin))) 
            
          
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Extract left and right line pixel positions
        self.x['left'], self.x['right'] = nonzerox[left_lane_inds],  nonzerox[right_lane_inds] 
        self.y['left'], self.y['right'] = nonzeroy[left_lane_inds],  nonzeroy[right_lane_inds]
        
        # Get line status information: valid = [True/False]
        self._analyse_line_status(verbose=verbose)
        ploty  = self.ploty 
        ploty2 = self.ploty2   
        for key in self.current_fit:
            # Fit a second order polynomial to each
            if True:
                self._bounded_decrease_search_margin(key)
                
                self.current_fit[key]  = np.polyfit(self.y[key], self.x[key], 2)

                # polynomial coefficients of last n iterations
                self.recent_fit[key].append(self.current_fit[key]) 

                # average fit coefficients
                self.best_fit[key] = np.average(self.recent_fit[key], axis=0)

                # Generate x and y values for plotting    
                A,B,C = self.current_fit[key]
                self.current_xfitted[key] = A*ploty2 + B*ploty + C

                # x values of the last n fits of the line    
                self.recent_xfitted[key].append(self.current_xfitted[key])
                
                self.best_xfitted[key] = np.average(self.recent_xfitted[key], axis=0)
            elif self.line_valid[key] == False:
                self._bounded_increase_search_margin(key)
        self._calculate_curvature_radius()
        self._calculate_average_curvature_radius()
        self._calculate_vehicle_position()

    
        if verbose==2:
            plt.imshow(out_img)
            plt.plot(self.current_xfitted['left'], ploty, color='yellow')
            plt.plot(self.current_xfitted['right'], ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([self.current_xfitted['left']-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.current_xfitted['left']+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.current_xfitted['right']-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.current_xfitted['right']+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        
        # Draw area between the lines 
        lane_window_left  = np.array([np.transpose(np.vstack([self.current_xfitted['left'], ploty]))])
        lane_window_right = np.array([np.flipud(np.transpose(np.vstack([self.current_xfitted['right'], ploty])))])
        lane_window_pts   = np.hstack((lane_window_left, lane_window_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([lane_window_pts]), (0,255, 0))
        
        if verbose==0:
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        elif verbose==1:
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,0, 255))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,0, 255))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            fig = plt.figure(figsize=(12,12))
            plt.imshow(result)
            plt.plot(self.x['left'], self.y['left'], 'r.')
            plt.plot(self.x['right'], self.y['right'], 'r.')

            plt.plot(self.current_xfitted['left'], ploty, color='yellow')
            plt.plot(self.current_xfitted['right'], ploty, color='yellow')
            
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()
            
        return result
        
    
    def get_radius_annotations(self, verbose=0):
        if verbose==0:
            textfields = ("radius of lane curvature: {:.0f} meters".format(self.average_radius_of_curvature),
                         "vehicle position: {:.2f} meters".format(self.vehicle_position))
            textcoords = ((80, 30), (80, 70))
            
        elif verbose==1:
            textfields = ("left: {:.2f} m".format(self.radius_of_curvature['left']),
                         "right: {:.2f} m".format(self.radius_of_curvature['right']),
                         "radius of lane curvature: {:0f} meters".format(self.average_radius_of_curvature),
                         "vehicle position: {:.2f} meters".format(self.vehicle_position))
            textcoords = ((145, 700), (900, 700), (80, 30), (80, 70))

        return textfields, textcoords
        
        
    def _calculate_curvature_radius(self):
        fit = {key: []  for key in self.x}
        for key in self.x:
            y_max = np.max(self.y[key])
            # Fit new polynomials to x,y in world space
            fit[key] = np.polyfit(self.y[key] * self.ymeter_per_pixel, self.x[key] * self.xmeter_per_pixel, 2)
            # Calculate the new radii of curvature
            A, B, C = fit[key]
            self.radius_of_curvature[key] = ((1 + (2*A*y_max* self.ymeter_per_pixel + B)**2)**1.5) / np.absolute(2*A)
            
    def _calculate_average_curvature_radius(self):
        
        fit        = {key: [] for key in self.best_xfitted}
        avg_radius = {key: [] for key in self.best_xfitted}
        
        for key in self.best_xfitted:
            y_max = np.max(self.ploty)
            # Fit new polynomials to x,y in world space
            fit[key] = np.polyfit(self.ploty * self.ymeter_per_pixel, self.best_xfitted[key] * self.xmeter_per_pixel, 2)
            # Calculate the new radii of curvature
            A, B, C = fit[key]
            avg_radius[key] = ((1 + (2*A*y_max* self.ymeter_per_pixel + B)**2)**1.5) / np.absolute(2*A)
        self.average_radius_of_curvature = min(((avg_radius['left'] + avg_radius['right'])/2), 10000)
        
    def _calculate_vehicle_position(self):
        xlane = {key: [] for key in self.best_xfitted}
        xcenter = 640
        for key in self.best_xfitted:
            xlane[key] = self.best_xfitted[key][-1]
        
        self.vehicle_position = ((xlane['right']  - xlane['left']) - xcenter)*self.xmeter_per_pixel
        

            
