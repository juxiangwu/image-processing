import sys
import vtk

try:
    from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    print('Using PyQT4')
except ImportError:
    try:
        from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        print('Using PyQT5')
    except ImportError:
        sys.exit('No PyQt found in our system!')

class QVTKWidget(QVTKRenderWindowInteractor):

    def __init__(self, parent = None):
        QVTKRenderWindowInteractor.__init__(self, parent)
        self.GetRenderWindow().GetInteractor().SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
