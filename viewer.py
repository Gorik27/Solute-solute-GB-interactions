import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, QtWidgets, QtOpenGL
import pyqtgraph as pg
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from pyqtgraph import GraphicsScene
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDrag
import copy
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import PyAtomisticTools as pat


class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.system = pat.System()
        self.natoms, self.bounds, _ = self.system.read('minimize.dat')
        self.system.read_gb('GBs.txt')
        self.ids = self.system.ids
        self.pairs = []
        self.gb = self.system.gb
                
        self.rs = self.system.coords[self.gb]
        self.r0 = np.transpose(np.mean(self.bounds, axis=1))
        self.rs -= self.r0
        colors = np.array([[0,0,1,1]]*len(self.rs))
        selection_cutoff = 0.03

        self.widget = My_GLViewWidget(self.rs, colors, selection_cutoff, self.pairs)
        self.widget.setMinimumSize(640, 480)
        
        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.addWidget(self.widget)
        self.setLayout(mainLayout)
        

        
    
class My_GLViewWidget(gl.GLViewWidget):
    def __init__(self, points, colors, selection_cutoff, pairs):
        super().__init__()
        self.selected_pair_index = -1
        self.atoms = points
        self.natoms = len(points)
        self.atoms0 = copy.deepcopy(points)
        self.pairs = pairs
        self.npairs = len(pairs)
        print(pairs)
        self.colors = colors
        self.selection_cutoff = selection_cutoff
        self.plot = gl.GLScatterPlotItem()
        self.addItem(self.plot)
        self.dragging = False
        self.size = 0.5
        self.cluster = []
        self.plot.setData(pos=self.atoms, color=self.colors, size=self.size, pxMode=False)

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Tab:# to do!!!!!!!!!!
            if self.selected_pair_index == -1:
                self.selected_pair_index = 0
            elif self.selected_pair_index < self.npairs:
                self.selected_pair_index += 1
            else:
                self.selected_pair_index = 0
                
            self.select_pair()
        elif ev.key() == QtCore.Qt.Key_Q:
            self.select_cluster()
        return super().keyPressEvent(ev)
    
    def select_pair(self):# to do!!!!!!!!!!!!!!
        print(self.pairs[self.selected_pair_index])
        self.atoms = self.atoms0[*self.pairs[self.selected_pair_index]]
        self.plot.setData(pos=self.atoms)
        
    def select_cluster(self):
        cluster = self.atoms0[self.cluster]
        cutoff = 11
        selection = self.select_nearest(cluster, cutoff)
        self.atoms = self.atoms0[selection]
        self.atoms -= np.mean(self.atoms, axis=0)
        colors = copy.deepcopy(self.colors)
        for index in self.cluster:
            colors[index] = np.array([0,1,0,1])
        colors = colors[selection]
        
        self.plot.setData(pos=self.atoms, color=colors)
        
    def select_nearest(self, cluster, cutoff):
        center = np.mean(cluster, axis=0)
        distance = np.sum((self.atoms0-center)**2, axis=1)**0.5 
        # to do special calc for periodic boundaries!!!!!!!!!!!
        selection = (distance<cutoff)
        return selection
        
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
            self.mousePos = lpos
            view_w = self.width()
            view_h = self.height()
            mouse_x = self.mousePos.x()
            mouse_y = self.mousePos.y()
            m=self.projectionMatrix()*self.viewMatrix()
            m=np.array(m.data(), dtype=np.float32).reshape((4,4))
            one_mat = np.ones((self.atoms.shape[0], 1))
            points_m = np.concatenate((self.atoms, one_mat), axis=1)
            new=np.matmul(points_m, m)
            new[:,:3]=new[:,:3]/new[:,3].reshape(-1,1)
            new=new[:,:3]
            projected_array = np.zeros((new.shape[0], 2))
            projected_array[:, 0] = (new[:,0] + 1)/2
            projected_array[:, 1] =(-new[:,1] + 1)/2
            self.projected_array=copy.deepcopy(projected_array)
            self.projected_array[:,0]=self.projected_array[:,0]*view_w
            self.projected_array[:, 1] = self.projected_array[:, 1]*view_h
            projected_array[:, 0] = (projected_array[:, 0] -
                                     (mouse_x/view_w))
            projected_array[:, 1] = (projected_array[:, 1] -
                                     (mouse_y/view_h))
        
            distance_array = np.power(np.power(projected_array[:, 0], 2) +
                                      np.power(projected_array[:, 1], 2), 0.5)
            min_index = np.nanargmin(distance_array)
            dmin = distance_array[min_index]
            if dmin<self.selection_cutoff:
                #print(f'selected point: {self.atoms[min_index]}')
                colors = copy.deepcopy(self.colors)
                if QApplication.keyboardModifiers() == Qt.ControlModifier:
                    self.cluster.append(min_index)
                    for index in self.cluster:
                        colors[index] = np.array([0,1,0,1])
                else:      
                    self.cluster = []
                    colors[min_index] = np.array([1,0,0,1])
                    if not self.dragging:
                        self.dragging = True
                        self.last_pos = lpos
                        self.selected = min_index
                self.plot.setData(color=colors)
            else:
                self.cluster = []
                self.plot.setData(color=self.colors)
                #print('no selection')
        return super().mousePressEvent(ev)
                
    def mouseReleaseEvent(self, ev):
        if self.dragging:
            if ev.button() == QtCore.Qt.RightButton:
                self.dragging = False
                self.selected = None
        return super().mouseReleaseEvent(ev)
        
    def mouseMoveEvent(self, ev):
        if self.dragging:
            lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
            view_w = self.width()
            view_h = self.height()
            vec = self.cameraPosition()
            scale = 0.1*self.cameraParams()['distance']
            a = vec.x()
            b = vec.y()
            c = vec.z()
            vec = np.array([a,b,c])
            
            camera_x = np.cross(np.array([0,0,1]), vec)
            camera_x /= np.linalg.norm(camera_x)
            camera_y = -np.cross(vec, camera_x)
            camera_y /= np.linalg.norm(camera_y)
            dx = 10*(lpos.x() - self.last_pos.x())/view_w
            dy = 10*(lpos.y() - self.last_pos.y())/view_h
            self.last_pos = lpos
            self.atoms[self.selected] += (camera_x*dx + camera_y*dy)*scale
            self.plot.setData(pos=self.atoms)
        return super().mouseMoveEvent(ev)



if __name__ == '__main__':    
    app = QtWidgets.QApplication(sys.argv)    
    pg.setConfigOptions(useOpenGL=True)
    Form = QtWidgets.QMainWindow()
    ui = MainWindow()    
    ui.show()    
    sys.exit(app.exec_())