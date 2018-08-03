#coding:utf-8
import vtk
from math import sqrt, exp, pi

edge = 20.0
box = (edge, edge, edge)
density = 20
step = edge / density
npoints = density**3



def f(x, y, z, e, n, l):
	r = sqrt(x**2 + y**2 + z**2)
	if abs(r) < 1e-10: return 0.0
	if l == 1:
		return 0.5 * sqrt(3.0/pi) * z/r * r**n * exp( -e * r)
	elif l == 2:
		return 0.25 * sqrt(5.0/pi) * (3*(z/r)**2 - 1) * r**n * exp( -e * r)


grid = vtk.vtkImageData()
grid.SetDimensions(density, density, density)
grid.SetOrigin(-box[0]/2.0, -box[1]/2.0, -box[2]/2.0)
grid.SetSpacing(box[0]/density, box[1]/density, box[2]/density)

data = vtk.vtkFloatArray()
data.SetNumberOfValues(npoints)

for i in range(density):
	x = box[0]/density*i - box[0]/2.0
	for j in range(density):
		y = box[1]/density*j - box[1]/2.0
		for k in range(density):
			z = box[2]/density*k - box[2]/2.0
			n = k + j*density + i*density*density
			data.SetValue(n, f(x, y, z, 0.5, 1, 2))

grid.GetPointData().SetScalars(data)

contour = vtk.vtkContourFilter()
contour.SetInputData(grid)
contour.SetValue(0, -0.1)
contour.SetValue(1,  0.1)

plane = vtk.vtkPlaneSource()
plane.SetResolution(100, 100)

t = vtk.vtkTransform()
t.Scale(box[0]-step, box[1]-step, box[2]-step)
t.Translate(-0.5/density, -0.5/density, -0.5/density)

trans = vtk.vtkTransformPolyDataFilter()
trans.SetTransform(t)
trans.SetInputConnection(plane.GetOutputPort())

probe = vtk.vtkProbeFilter()
probe.SetInputConnection(trans.GetOutputPort())
probe.SetSourceData(grid)

outline = vtk.vtkOutlineFilter()
outline.SetInputData(grid)

contour_lut = vtk.vtkLookupTable()
contour_lut.SetTableRange(-0.15, 0.15)

# Update probe filter to get valid scalar range
probe.Update()
srange = probe.GetOutput().GetScalarRange()
# print "Probe range =", srange

probe_lut = vtk.vtkLookupTable()
probe_lut.SetTableRange(srange)
probe_lut.SetNumberOfTableValues(255)
for i in range(255):
	probe_lut.SetTableValue(i, i/255.0, 0.0, 1 - i/255.0, 1.0)

contour_mapper = vtk.vtkPolyDataMapper()
contour_mapper.SetInputConnection(contour.GetOutputPort())
contour_mapper.SetLookupTable(contour_lut)
contour_mapper.UseLookupTableScalarRangeOn()

planeMapper = vtk.vtkPolyDataMapper()
planeMapper.SetInputConnection(probe.GetOutputPort())
planeMapper.UseLookupTableScalarRangeOn()
planeMapper.SetLookupTable(probe_lut)

outline_mapper = vtk.vtkPolyDataMapper()
outline_mapper.SetInputConnection(outline.GetOutputPort())

contour_actor = vtk.vtkActor()
contour_actor.SetMapper(contour_mapper)
contour_actor.GetProperty().SetRepresentationToWireframe()

outline_actor = vtk.vtkActor()
outline_actor.SetMapper(outline_mapper)

planeActor = vtk.vtkActor()
planeActor.SetMapper(planeMapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(contour_actor)
renderer.AddActor(outline_actor)
renderer.AddActor(planeActor)
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