#coding:utf-8
import vtk

box = (5.0, 5.0, 5.0)
density = 15
npoints = density**3
spacing = (box[0]/(density-1), box[1]/(density-1), box[2]/(density-1))



def f(x, y, z):
	return (x**2 + y**2) * z**2


grid = vtk.vtkImageData()
grid.SetDimensions(density, density, density)
grid.SetOrigin(-box[0]/2.0, -box[1]/2.0, -box[2]/2.0)
grid.SetSpacing(spacing[0], spacing[1], spacing[2])

data = vtk.vtkFloatArray()
data.SetNumberOfValues(npoints)

for i in range(density):
	x = spacing[0]*i - box[0]/2.0
	for j in range(density):
		y = spacing[1]*j - box[1]/2.0
		for k in range(density):
			z = spacing[2]*k - box[2]/2.0
			n = k + j*density + i*density*density
			data.SetValue(n, f(x, y, z))

grid.GetPointData().SetScalars(data)

grad = vtk.vtkImageGradient()
grad.SetDimensionality(3)
grad.SetInputData(grid)

attrib = vtk.vtkAssignAttribute()
attrib.SetInputConnection(grad.GetOutputPort())
attrib.Assign(vtk.vtkDataSetAttributes.SCALARS, vtk.vtkDataSetAttributes.VECTORS, vtk.vtkAssignAttribute.POINT_DATA)

contour = vtk.vtkContourFilter()
contour.SetInputData(grid)
contour.SetValue(0, 4.0)

hog = vtk.vtkHedgeHog()
hog.SetInputConnection(attrib.GetOutputPort())
hog.SetScaleFactor(0.01)

outline = vtk.vtkOutlineFilter()
outline.SetInputConnection(contour.GetOutputPort())

contour_mapper = vtk.vtkPolyDataMapper()
contour_mapper.SetInputConnection(contour.GetOutputPort())

outline_mapper = vtk.vtkPolyDataMapper()
outline_mapper.SetInputConnection(outline.GetOutputPort())

hog_mapper = vtk.vtkPolyDataMapper()
hog_mapper.SetInputConnection(hog.GetOutputPort())

contour_actor = vtk.vtkActor()
contour_actor.SetMapper(contour_mapper)

outline_actor = vtk.vtkActor()
outline_actor.SetMapper(outline_mapper)

hog_actor = vtk.vtkActor()
hog_actor.SetMapper(hog_mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(contour_actor)
renderer.AddActor(hog_actor)
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