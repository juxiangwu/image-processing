#coding:utf-8
import vtk
box = (5.0, 5.0, 5.0)
density = 50
npoints = density**3



def f(x, y, z):
	return (x**2 + y**2) * z**2


grid = vtk.vtkImageData()
grid.SetDimensions(density, density, density)
grid.SetOrigin(-box[0]/2.0, -box[1]/2.0, -box[2]/2.0)
grid.SetSpacing(box[0]/density, box[1]/density, box[2]/density)

# print grid
# print "scalar range =", grid.GetScalarRange()

data = vtk.vtkFloatArray()
data.SetNumberOfValues(npoints)

for i in range(density):
	z = box[2]/density*i - box[2]/2.0
	for j in range(density):
		y = box[1]/density*j - box[1]/2.0
		for k in range(density):
			x = box[0]/density*k - box[0]/2.0
			n = k + j*density + i*density*density
			data.SetValue(n, f(x, y, z))

grid.GetPointData().SetScalars(data)

# print grid
# print "scalar range =", grid.GetScalarRange()

contour = vtk.vtkContourFilter()
contour.SetInputData(grid)
contour.SetValue(0,  0.1)
contour.SetValue(1,  1.0)
contour.SetValue(2,  5.0)
contour.SetValue(3, 10.0)

outline = vtk.vtkOutlineFilter()
outline.SetInputConnection(contour.GetOutputPort())

lut = vtk.vtkLookupTable()
lut.SetTableRange(0.0, 10.0)

contour_mapper = vtk.vtkPolyDataMapper()
contour_mapper.SetInputConnection(contour.GetOutputPort())
contour_mapper.SetLookupTable(lut)
contour_mapper.UseLookupTableScalarRangeOn()

outline_mapper = vtk.vtkPolyDataMapper()
outline_mapper.SetInputConnection(outline.GetOutputPort())

contour_actor = vtk.vtkActor()
contour_actor.SetMapper(contour_mapper)

outline_actor = vtk.vtkActor()
outline_actor.SetMapper(outline_mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(contour_actor)
renderer.AddActor(outline_actor)
renderer.ResetCamera()

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(800,600)

istyle = vtk.vtkInteractorStyleSwitch()
istyle.SetCurrentStyleToTrackballCamera()

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
interactor.SetInteractorStyle(istyle)

interactor.Initialize()
interactor.Start()