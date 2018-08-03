#coding:utf-8
import vtk

box = (5.0, 5.0, 5.0)
density = 11
npoints = density**3
spacing = (box[0]/(density-1), box[1]/(density-1), box[2]/(density-1))

def f(x, y, z):
	return (x**2 + y**2) * z**2

def fprime(x, y, z):
	#
	# Gradient of (x^2 + y^2) z^2
	#
	ddx = z**2 * 2*x
	ddy = z**2 * 2*y
	ddz = (x**2 + y**2) * 2*z
	return ddz, ddy, ddx


grid = vtk.vtkImageData()
grid.SetDimensions(density, density, density)
grid.SetOrigin(-box[0]/2.0, -box[1]/2.0, -box[2]/2.0)
grid.SetSpacing(spacing[0], spacing[1], spacing[2])

scalar_data = vtk.vtkFloatArray()
scalar_data.SetNumberOfValues(npoints)

vector_data = vtk.vtkFloatArray()
vector_data.SetNumberOfComponents(3)
vector_data.SetNumberOfTuples(npoints)

for i in range(density):
	x = spacing[0]*i - box[0]/2.0
	for j in range(density):
		y = spacing[1]*j - box[1]/2.0
		for k in range(density):
			z = spacing[2]*k - box[2]/2.0
			n = k + j*density + i*density*density
			scalar_data.SetValue(n, f(x, y, z))
			vector_data.SetTuple(n, fprime(x, y, z))

grid.GetPointData().SetScalars(scalar_data)
grid.GetPointData().SetVectors(vector_data)

contour = vtk.vtkContourFilter()
contour.SetInputData(grid)
contour.SetValue(0, 4.0)

lut = vtk.vtkLookupTable()
lut.SetTableRange(0.0, 10.0)

contour_mapper = vtk.vtkPolyDataMapper()
contour_mapper.SetInputConnection(contour.GetOutputPort())
contour_mapper.SetLookupTable(lut)
contour_mapper.UseLookupTableScalarRangeOn()

contour_actor = vtk.vtkActor()
contour_actor.SetMapper(contour_mapper)

hog = vtk.vtkHedgeHog()
hog.SetInputData(grid)
hog.SetScaleFactor(0.01)

hog_mapper = vtk.vtkPolyDataMapper()
hog_mapper.SetInputConnection(hog.GetOutputPort())
hog_mapper.SetScalarModeToUsePointData()

hog_actor = vtk.vtkActor()
hog_actor.SetMapper(hog_mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(contour_actor)
renderer.AddActor(hog_actor)
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