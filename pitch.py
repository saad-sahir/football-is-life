import matplotlib.pyplot as plt
import numpy as np

class Pitch:
    def __init__(self, length = 1.1, width = 0.7, constants = (
            (0.18, 0.12),
            (0.08, 0.21),
            (0.28)
        )):
        self.length = length
        self.width = width

        self.pb = constants[0] # penalty box constant
        self.gb = constants[1] # goalbox constant
        self.gp = constants[2] # goalpost constant

        self.window = (
            (-self.length/5, self.length*1.2),
            (-self.width/5, self.width*1.2)
        )

        self.corners = self._corners()
        self.boundaries = self._boundaries()
        self.halfpoint = self._halfpoint()
        self.penalty_boxes = self._penalty_box()
        self.goal_boxes = self._goal_box()

        self.lines = self._lines()
        self.points = self._points()
        

    def _corners(self):
        return {
            'bl':(0,0),
            'tl':(0,self.width),
            'br':(self.length, 0),
            'tr':(self.length, self.width),
        }
       
    def _boundaries(self):
        return {
            'home_gl': (self.corners['bl'], self.corners['tl']),
            'away_gl': (self.corners['br'], self.corners['tr']),
            'cti': (self.corners['bl'], self.corners['br']), # close throw in
            'fti': (self.corners['tl'], self.corners['tr']), # far throw in
        }
    
    def _halfpoint(self):
        return {
            'fhl' : (self.length/2, self.width),
            'chl' : (self.length/2, 0),
        }
    
    def _penalty_box(self):
        return {
            # home
            'hcgl' : (0, self.corners['bl'][1]+self.pb[1]), # home close (penalty box) goal line point 
            'hfgl' : (0, self.corners['tl'][1]-self.pb[1]), # home far (penalty box) goal line point 
            'hfpc' : (self.pb[0], self.corners['tl'][1]-self.pb[1]), # home far penalty box corner
            'hcpc' : (self.pb[0], self.corners['bl'][1]+self.pb[1]), # home close penalty box corner
            # away
            'acgl' : (self.length, self.corners['br'][1]+self.pb[1]), # away close (penalty box) goal line point
            'afgl' : (self.length, self.corners['tr'][1]-self.pb[1]), # away far (penalty box) goal line point
            'afpc' : (self.length - self.pb[0], self.corners['tr'][1]-self.pb[1]), # away far penalty box corner
            'acpc' : (self.length - self.pb[0], self.corners['br'][1]+self.pb[1]), # away close penalty box corner
        }
    
    def _goal_box(self):
        return {
            # home
            'hcgb' : (0, self.corners['bl'][1]+self.gb[1]), # home close goal box point 
            'hfgb' : (0, self.corners['tl'][1]-self.gb[1]), # home far goal box point
            'hfgc' : (self.gb[0], self.corners['tl'][1]-self.gb[1]), # far goal box corner
            'hcgc' : (self.gb[0], self.corners['bl'][1]+self.gb[1]), # close goal box corner
            'hcgp': (0, self.corners['bl'][1]+self.gp), # home close goal post 
            'hfgp': (0, self.corners['tl'][1]-self.gp), # home far goal post
            # away
            'acgb' : (self.length, self.corners['br'][1]+self.gb[1]), # away close goal box point
            'afgb' : (self.length, self.corners['tr'][1]-self.gb[1]), # away far goal box point
            'afgc' : (self.length - self.gb[0], self.corners['tr'][1]-self.gb[1]), # away far goal box corner
            'acgc' : (self.length - self.gb[0], self.corners['br'][1]+self.gb[1]), # away lose goal box corner
            'acgp': (self.length, self.corners['br'][1]+self.gp), # away close goal post
            'afgp': (self.length, self.corners['tr'][1]-self.gp), # away far goal post
      
        }   

    def _lines(self):
        return {
            **self._boundaries(),
            # halfline
            'hfl' : (self.halfpoint['fhl'], self.halfpoint['chl']),
            # home penalty box
            'hfypb': (self.penalty_boxes['hfgl'], self.penalty_boxes['hfpc']),
            'hcypb': (self.penalty_boxes['hcgl'], self.penalty_boxes['hcpc']),
            'hxpb': (self.penalty_boxes['hfpc'], self.penalty_boxes['hcpc']),
            # away penalty box
            'afypb': (self.penalty_boxes['afgl'], self.penalty_boxes['afpc']),
            'acypb': (self.penalty_boxes['acgl'], self.penalty_boxes['acpc']),
            'axpb': (self.penalty_boxes['afpc'], self.penalty_boxes['acpc']), 
            # home goal box
            'hfygb': (self.goal_boxes['hfgb'], self.goal_boxes['hfgc']),
            'hcygb': (self.goal_boxes['hcgb'], self.goal_boxes['hcgc']),
            'hxgb': (self.goal_boxes['hcgc'], self.goal_boxes['hfgc']),
            # away goal box
            'afygb': (self.goal_boxes['afgb'], self.goal_boxes['afgc']),
            'acygb': (self.goal_boxes['acgb'], self.goal_boxes['acgc']),
            'axgb': (self.goal_boxes['acgc'], self.goal_boxes['afgc']),
        }
    
    def _points(self):
        return {
            'ko':(self.length/2, self.width/2),
            **self._corners(),
            **self._halfpoint(),
            **self._penalty_box(),
            **self._goal_box(),
        }
    
    def _angle_between(self, line1, line2):
        '''
        Calculate the angle in degrees between two lines given by two points each.
        '''
        def unit_vector(line):
            ''' Create a unit vector for a line. '''
            p1, p2 = line
            vector = np.subtract(p2, p1)
            norm = np.linalg.norm(vector)
            return vector / norm

        v1 = unit_vector(line1)
        v2 = unit_vector(line2)
        dot_product = np.dot(v1, v2)
        angle = np.arccos(dot_product)
        return np.degrees(angle)

    def _find_parallel_lines(self):
        '''
        Find pairs of parallel lines on the pitch.
        '''
        parallels = []
        for line_name1, line1 in self.lines.items():
            for line_name2, line2 in self.lines.items():
                if line_name1 < line_name2:  # Avoid repeating the same pair
                    angle = self._angle_between_lines(line1, line2)
                    if np.isclose(angle, 0) or np.isclose(angle, 180):
                        parallels.append((line_name1, line_name2))
        return parallels
    
    def _find_perpendicular_lines(self):
        '''
        Find pairs of perpendicular lines on the pitch.
        '''
        perpendiculars = []
        for line_name1, line1 in self.lines.items():
            for line_name2, line2 in self.lines.items():
                if line_name1 < line_name2:  # Avoid repeating the same pair
                    angle = self._angle_between_lines(line1, line2)
                    if np.isclose(angle, 90):
                        perpendiculars.append((line_name1, line_name2))
        return perpendiculars

    def draw(self, lines, points):
        fig, ax = plt.subplots()

        for line in lines.values():
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color='green')

        for point in points.values():
            ax.plot(point[0], point[1], 'ro')

        # Set the limits of the plot
        ax.set_xlim(self.window[0])
        ax.set_ylim(self.window[1])
        ax.set_aspect('equal')  # This ensures the pitch is drawn to scale

        plt.show()

if __name__ == "__main__":
    pitch = Pitch()
    pitch.draw(pitch.lines, pitch.points)