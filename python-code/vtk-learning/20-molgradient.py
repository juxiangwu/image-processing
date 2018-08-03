#coding:utf-8
import vtk
from sys import argv
import numpy as N


def read_xyz(fnm):

	out = {}

	f = open(fnm)

	l = f.readline()
	n = int(l)

	sym = []
	coord = []
	grad = []

	l = f.readline()
	for i in range(n):
		l = f.readline().split()
		sym.append(int(l[1]))
		coord.append((lambda x: float(x)/0.529177209, l[2:5]))
		grad.append((l[5:8]))

	return sym, N.array(coord), N.array(grad)


symbols, coordinates, gradients = read_xyz(argv[1])
nat = len(symbols)

molecule = vtk.vtkMolecule()

for i in range(nat):
	s = symbols[i]
	a = coordinates[i]
	molecule.AppendAtom(s, a[0], a[1], a[2])

for i in range(nat):
	a = coordinates[i]
	for j in range(i+1, nat):
		b = coordinates[j]
		d = N.sqrt(N.sum((a-b)**2))
		if d < 3.0:
			molecule.AppendBond(i, j, 1)

atom_points = vtk.vtkPoints()
atom_points.SetNumberOfPoints(nat)

vectors = vtk.vtkFloatArray()
vectors.SetNumberOfComponents(3)
vectors.SetNumberOfTuples(nat)

for i in range(nat):
	a = coordinates[i]
	atom_points.InsertPoint(i, a[0], a[1], a[2])
	vectors.SetTuple(i, gradients[i])

grid = vtk.vtkUnstructuredGrid()
grid.SetPoints(atom_points)
grid.GetPointData().SetVectors(vectors)

molMapper = vtk.vtkMoleculeMapper()
molMapper.SetInputData(molecule)

molActor = vtk.vtkActor()
molActor.SetMapper(molMapper)

arrow = vtk.vtkArrowSource()

glyph = vtk.vtkGlyph3D()
glyph.SetInputData(grid)
glyph.SetSourceConnection(arrow.GetOutputPort())
glyph.SetScaleModeToScaleByVector()
glyph.SetScaleFactor(30.0)

glyphMapper = vtk.vtkPolyDataMapper()
glyphMapper.SetInputConnection(glyph.GetOutputPort())
glyphMapper.SetScalarModeToUsePointData()

glyphActor = vtk.vtkActor()
glyphActor.SetMapper(glyphMapper)
glyphActor.GetProperty().SetColor(1.0, 1.0, 0.0)

renderer = vtk.vtkRenderer()
renderer.AddActor(molActor)
renderer.AddActor(glyphActor)
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
