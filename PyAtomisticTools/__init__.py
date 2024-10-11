import numpy as np
import re
#from numba import njit
import copy
from subprocess import Popen
eV2kJmole = 96.4915666370759

class System:
    def __init__(self, data=None, bounds=None, filename=None, only_coords=True):
        if filename:
            self.read(filename, only_coords)
        else:
            self.filename = None
            self.system = data    
            if data:
                self.natoms = len(data)
                self.initialized = True
            else:
                self.natoms = None
                self.initialized = False
            self.bounds = bounds
            self.gb_initialized = False
        
            
    def calc_int(self, id1, id2):
        structure = 'tmp.dat'
        self.write(structure)
            
        script = 'in.int'
        task1 = f'lmp -in {script} -var structure {structure} -var id1 {id1} -var id2 {id2}'
        script = 'in.seg'
        task2 = f'lmp -in {script} -var structure {structure} -var id {id1}'
        task3 = f'lmp -in {script} -var structure {structure} -var id {id2}'
        script = 'in.pure'
        task4 = f'lmp -in {script} -var structure {structure}'
        tasks = [task1, task2, task3, task4]
        processes = []
        i = 0
        for task in tasks:
            fname = f"tmp_out{i}"
            with open(fname, 'w') as f:
                i+=1
                p = Popen(task.split(),stdout=f)
            processes.append((p, fname))
            
        es = []
        for p, fname in processes:
            p.wait()
            with open(fname) as f:
                for line in f:
                    e_pattern = "Seg energy "
                    if e_pattern in line:
                        energy = float(line.replace(e_pattern, '').replace('\n', ''))
                        es.append(energy)
        print(es)
        self.int = (es[0] - es[1] - es[2] + es[3])*eV2kJmole
        return self.int
        
    @property
    def coords(self):
        if self.initialized:
            return copy.deepcopy(self.system[:, 2:5])
        else:
            raise AttributeError("system hasn't been set!")
            
    def move_atom(self, id, dr):
        self.system[id, 2:5] += dr
        
    @property
    def ids(self):
        if self.initialized:
            return copy.deepcopy(self.system[:, 0].astype(int))
        else:
            raise AttributeError("system hasn't been set!")
            
    def read_gb(self, filename):
        data = np.loadtxt(filename, skiprows=1)
        self.gb_initialized = True
        self._gb_ids = data[:,0]
        self._gb = np.isin(self.ids, self._gb_ids)
    
    @property
    def gb_ids(self):
        if self.initialized:
            if self.gb_initialized:
                return copy.deepcopy(self._gb_ids)
            else:
                AttributeError("GBs hasn't been set!")
        else:
            raise AttributeError("system hasn't been set!")
            
    @property
    def gb(self):
        if self.initialized:
            if self.gb_initialized:
                return copy.deepcopy(self._gb)
            else:
                AttributeError("GBs hasn't been set!")
        else:
            raise AttributeError("system hasn't been set!")
        
            
    def read(self, filename, only_coords=True):
        """
        Read LAMMPS dat file
        
        Parameters
        ----------
        filename : str
            name of dat file.
    
        only_coords : bool (True)
            Return only 5 columns of data (id, type, x, y, z).
            
        Returns
        -------
        Natoms : int
            DNumber of atoms.
        bounds : ndarray shape=(3,2)
            [[xlo, xhi], [ylo, yhi], [zlo, zhi]].
        data : ndarray shape=(Natoms, 5 or any if only_coords=False)
            atoms properties.
    
        """
        self.filename = filename
        with open(filename) as f:
            i = 0
            bounds = np.zeros((3,2))
            break_flag = False
            for line in f:
                if re.match(r'^[0-9]+ [0-9]+ [0-9]+\.[0-9]+ [0-9]+\.[0-9]+ [0-9]+\.[0-9]+', line):
                    break_flag = True
                    break
                else: i += 1
                
                if re.match(r'^[0-9]+ atoms$', line):
                    natoms = int(line.split(' ')[0])
                elif re.match(r'^[+-]?[0-9]+\.[0-9]+ [+-]?[0-9]+\.[0-9]+ [xyz]lo [xyz]hi', line):
                    for j, ksi in enumerate(['x', 'y', 'z']):
                        if re.search(f'{ksi}lo', line):
                            args = line.split(' ')[:2]
                            bounds[j,0] = float(args[0])
                            bounds[j,1] = float(args[1])
                            break
            if not break_flag:
                raise ValueError('Cannot read file of given format')
                return
              
        data = np.loadtxt(filename, skiprows=i, max_rows=natoms)
        if only_coords:
            data = data[:,:5]
        #print(data[0], data[-1])
        self.natoms = natoms
        self.bounds = bounds
        self.system = data
        self.initialized = True
        return natoms, bounds, data
    
    def write(self, filename):
        out = f"""LAMMPS data file via PyAtomisticTool

{self.natoms} atoms
1 atom types

{self.bounds[0,0]} {self.bounds[0,1]} xlo xhi
{self.bounds[1,0]} {self.bounds[1,1]} ylo yhi
{self.bounds[2,0]} {self.bounds[2,1]} zlo zhi

Masses

1 107.8682

Atoms # atomic

"""
        
        with open(filename, 'w') as f:
            f.write(out)
            fmt = ['%d', '%d']
            for i in range(self.system.shape[1]-2):
                fmt.append('%.18f')
            np.savetxt(f, self.system, fmt=fmt)
        
    

# def read_neigbors_file(filename, ids):
#     with open(filename) as f:
#         text = f.read()
#     lines = text.split('\n')
#     return _read_neigbors_file(ids, lines)
 
# @njit
# def _read_neigbors_file(ids, lines):
#     pairs = []
#     i = 0
#     for line in lines:
#         print(i)
#         if i == 0:
#             i += 1
#             continue
#         i += 1
#         args = line.replace('\n', '').split(' ')
#         if len(args)>1:
#             i0 = np.where(ids==int(args[0]))[0]
            
#             #print(line)
#             #print(i0)
#             for id in args[2:]:
#                 if id != '':
#                     i1 = np.where(ids==int(id))[0]
#                     #print(i1)
#                     pairs.append((i0, i1))
                
#     return pairs

# s = System()
# s.read('../SSint/minimize.dat')

# data = read_neigbors_file('../SSint/neigbors.txt', s.ids)

# s = System(filename='../SSint/minimize.dat')
# s.calc_int(1,2)
