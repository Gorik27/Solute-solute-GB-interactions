import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtGui, uic
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, QtWidgets, QtOpenGL
import pyqtgraph as pg
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from pyqtgraph import GraphicsScene
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QPushButton, \
QFileDialog, QInputDialog
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDrag
import copy
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import PyAtomisticTools as pat


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        uic.loadUi('viewer.ui', self)
        self.system = pat.System()
        self.natoms, self.bounds, _ = self.system.read('minimize.dat')
        self.system.read_gb('GBs.txt')
        self.only_gb = True
        self.actionShow_Hide_non_GB_atoms.setChecked(True) 
        self.pairs = []
        self.gb = self.system.gb
                
        self.rs = self.system.coords[self.gb]
        self.r0 = np.transpose(np.mean(self.bounds, axis=1))
        self.rs -= self.r0
        self.ids = self.system.ids[self.gb]
        self.colors = np.array([[0,0,1,1]]*len(self.rs))
        self.selection_cutoff = 0.03
        self.neighbors_cutoff = 11

        self.GLwidget = My_GLViewWidget(self.rs, self.colors, 
                                        self.selection_cutoff, self.pairs, self)
        self.GLwidget.setMinimumSize(640, 480)
        
        self.GLlayout.addWidget(self.GLwidget)
        self.text_output.setText('Hello world!')
        
        self.actionOpen.triggered.connect(self.open_dat)
        self.actionOpen_GBs.triggered.connect(self.open_gb)
        self.actionShow_Hide_non_GB_atoms.triggered.connect(self.switch_gb)
        self.actionNeighbors_cutoff.triggered.connect(self.set_neigbors_cutoff)
        
        
    def set_neigbors_cutoff(self):
        text, ok = QInputDialog.getText(self, 'input dialog', 'Neighbors cutoff (A)')
        if ok:
            self.neighbors_cutoff = float(text)
            self.GLwidget.update_cutoff()
        
    def open_dat(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
                                            '.',"LAMMPS data (*.dat *.txt)")
        if fname[0]:
            self.filename_dat = fname[0]
            self.system = pat.System()
            self.natoms, self.bounds, _ = self.system.read(self.filename_dat)
            self.rs = self.system.coords
            self.ids = self.system.ids
            self.r0 = np.transpose(np.mean(self.bounds, axis=1))
            self.rs -= self.r0
            self.colors = np.array([[0,0,1,1]]*len(self.rs))
            self.gb = None
            self.only_gb = False
            self.actionShow_Hide_non_GB_atoms.setChecked(False) 
            self.GLwidget.myreload(self.rs, self.colors, 
                                 self.selection_cutoff, self.pairs)
            
    def open_gb(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
                                            '.',"GBs ids (*.txt)")
        if fname[0]:
            self.filename_gb = fname[0]
            self.system.read_gb('GBs.txt')
            self.gb = self.system.gb
            
    def switch_gb(self):
        if self.only_gb:
            self.only_gb = False
            self.rs = self.system.coords
            self.ids = self.system.ids
            self.r0 = np.transpose(np.mean(self.bounds, axis=1))
            self.rs -= self.r0
            self.colors = np.array([[0,0,1,1]]*len(self.rs))
            self.GLwidget.myreload(self.rs, self.colors, 
                                 self.selection_cutoff, self.pairs)
        else:
            if not np.any(self.gb):
                self.actionShow_Hide_non_GB_atoms.setChecked(False) 
                raise ValueError('GBs file is not loaded!')
            self.only_gb = True
            self.rs = self.system.coords[self.gb]
            self.ids = self.system.ids[self.gb]
            self.r0 = np.transpose(np.mean(self.bounds, axis=1))
            self.rs -= self.r0
            self.colors = np.array([[0,0,1,1]]*len(self.rs))
            self.GLwidget.myreload(self.rs, self.colors, 
                                 self.selection_cutoff, self.pairs)

class My_GLViewWidget(gl.GLViewWidget):
    def __init__(self, points, colors, selection_cutoff, pairs, parent):
        super().__init__()
        self.selected_pair_index = -1
        self.parent = parent
        self.atoms = points
        self.natoms = len(points)
        self.atoms0 = copy.deepcopy(points)
        self.pairs = pairs
        self.npairs = len(pairs)
        self.colors = colors
        self.selection_cutoff = selection_cutoff
        self.plot = gl.GLScatterPlotItem()
        self.addItem(self.plot)
        self.dragging = False
        self.size = 0.5
        self.cluster = []
        self.cluster_view = False
        self.previous_pos = None
        self.plot.setData(pos=self.atoms, color=self.colors, size=self.size, 
                          pxMode=False)
        self.ids = copy.deepcopy(self.parent.ids)
        self.ids0 = copy.deepcopy(self.parent.ids)
        self.selected = None
        
    def myreload(self, points, colors, selection_cutoff, pairs):
        self.clear()
        self.selected_pair_index = -1
        self.atoms = points
        self.natoms = len(points)
        self.atoms0 = copy.deepcopy(points)
        self.pairs = pairs
        self.npairs = len(pairs)
        self.colors = colors
        self.selection_cutoff = selection_cutoff
        self.plot = gl.GLScatterPlotItem()
        self.addItem(self.plot)
        self.dragging = False
        self.size = 0.5
        self.cluster = []
        self.cluster_view = False
        self.previous_pos = None
        self.plot.setData(pos=self.atoms, color=self.colors, size=self.size, 
                          pxMode=False)

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
        elif ev.key() == QtCore.Qt.Key_R:
            self.full_view()
        elif ev.key() == QtCore.Qt.Key_O:
            self.reset_origin()
        elif ev.key() == (Qt.Key_Control and Qt.Key_Z):
            return self.undo_movement()
        return super().keyPressEvent(ev)
    
    def select_pair(self):# to do!!!!!!!!!!!!!!
        print(self.pairs[self.selected_pair_index])
        self.atoms = self.atoms0[*self.pairs[self.selected_pair_index]]
        self.plot.setData(pos=self.atoms)
        
    def select_cluster(self):
        if self.cluster:
            self.cluster_view = True
            cluster = self.atoms0[self.cluster]
            selection = self.select_nearest(cluster)
            self.atoms = self.atoms0[selection]
            self.ids = self.ids0[selection]
            self.last_offset = np.mean(self.atoms, axis=0)
            self.atoms -= self.last_offset
            colors = copy.deepcopy(self.colors)
            for index in self.cluster:
                colors[index] = np.array([0,1,0,1])
            colors = colors[selection]
            self.plot.setData(pos=self.atoms, color=colors)
        
    def select_nearest(self, cluster):
        center = np.mean(cluster, axis=0)
        distance = np.sum((self.atoms0-center)**2, axis=1)**0.5 
        # to do special calc for periodic boundaries!!!!!!!!!!!
        selection = (distance<self.parent.neighbors_cutoff)
        return selection
    
    def update_cutoff(self):
        if self.cluster_view:
            self.select_cluster()
            self.update_text()
            
    def update_text(self):
        cluster_ids = []
        for ind in self.cluster:
            cluster_ids.append(self.ids0[ind])
        if self.selected:
            selected_id = self.ids0[self.selected]
        else:
            selected_id = None
        text = f"""viewer output:

Neighbors cutoff {self.parent.neighbors_cutoff}

Interaction energy {0}

Selected cluster {cluster_ids}

Moved atom {selected_id}
"""
        self.parent.text_output.setText(text)
        
    def full_view(self):
        if self.cluster_view:
            colors = copy.deepcopy(self.colors)
            for index in self.cluster:
                colors[index] = np.array([0,1,0,1])
            self.plot.setData(pos=self.atoms0-self.last_offset, color=colors)
            self.cluster_view = False
        
    def reset_origin(self):
        if not self.cluster_view:
            self.plot.setData(pos=self.atoms0)
        
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
                if QApplication.keyboardModifiers() == Qt.ControlModifier:
                    self.cluster_selection(min_index)
                else:      
                    self.moving_selection(min_index, lpos)
            else:
                self.cluster = []
                self.plot.setData(color=self.colors)
                self.update_text()
                #print('no selection')
        return super().mousePressEvent(ev)
    
    def cluster_selection(self, min_index):
        if min_index in self.cluster:
            return
        colors = copy.deepcopy(self.colors)
        self.cluster.append(min_index)
        for index in self.cluster:
            colors[index] = np.array([0,1,0,1])
        self.plot.setData(color=colors)
        self.update_text()
        
    def moving_selection(self, min_index, lpos):
        colors = copy.deepcopy(self.colors)
        self.cluster = []
        colors[min_index] = np.array([1,0,0,1])
        if not self.dragging:
            self.dragging = True
            self.last_pos = lpos
            self.previous_pos = lpos # for undo
            self.selected = min_index
            self.update_text()
        self.plot.setData(color=colors)
                
    def mouseReleaseEvent(self, ev):
        if self.dragging:
            if ev.button() == QtCore.Qt.RightButton:
                self.dragging = False
                self.selected = None
                self.update_text()
        return super().mouseReleaseEvent(ev)
        
    def mouseMoveEvent(self, ev):
        if self.dragging:
            lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
            view_w = self.width()
            view_h = self.height()
            vec = self.cameraPosition()
            scale = self.cameraParams()['distance']
            self.last_scale = scale
            a = vec.x()
            b = vec.y()
            c = vec.z()
            vec = np.array([a,b,c])
            
            camera_x = np.cross(np.array([0,0,1]), vec)
            camera_x /= np.linalg.norm(camera_x)
            camera_y = -np.cross(vec, camera_x)
            camera_y /= np.linalg.norm(camera_y)
            self.last_camera_x = camera_x
            self.last_camera_y = camera_y
            dx = (lpos.x() - self.last_pos.x())/view_w
            dy = (lpos.y() - self.last_pos.y())/view_h
            self.last_pos = lpos
            self.atoms[self.selected] += (camera_x*dx + camera_y*dy)*scale
            self.plot.setData(pos=self.atoms)
            self.last_selected = copy.copy(self.selected)
        return super().mouseMoveEvent(ev)

    def undo_movement(self):
        if not self.previous_pos:
            return 
        view_w = self.width()
        view_h = self.height()
        scale = self.last_scale
        camera_x = self.last_camera_x
        camera_y = self.last_camera_y
        dx = (self.previous_pos.x() - self.last_pos.x())/view_w
        dy = (self.previous_pos.y() - self.last_pos.y())/view_h
        self.atoms[self.last_selected] += (camera_x*dx + camera_y*dy)*scale
        self.plot.setData(pos=self.atoms)
        self.previous_pos = None

if __name__ == '__main__':    
    app = QtWidgets.QApplication(sys.argv)    
    pg.setConfigOptions(useOpenGL=True)
    Form = QtWidgets.QMainWindow()
    ui = MainWindow()    
    ui.show()    
    sys.exit(app.exec_())