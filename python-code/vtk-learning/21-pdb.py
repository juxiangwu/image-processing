#coding:utf-8

import vtk
import numpy

symdict = { "N" : 7, "C" : 6, "O" : 8, "H" : 1 }
coords = []
symbols = []
pdbf = open("2EVQ.pdb")
line = pdbf.readline()
natoms = 0
while line[:6] != "MODEL ":
	line = pdbf.readline()
while line:
	modelno = int(line[6:])
	line = pdbf.readline()
	i = 0
	while line[:6] != "ENDMDL":
		sym = line[12:16].strip()
		if sym[:1] in ["N", "C", "O"]:
			if modelno == 1:
				coords.append([ [], [], [] ])
				natoms += 1
				symbols.append(symdict[sym[0]])
			x = float(line[30:38])
			y = float(line[38:46])
			z = float(line[46:54])
			coords[i][0].append(x)
			coords[i][1].append(y)
			coords[i][2].append(z)
			i += 1
		line = pdbf.readline()
	while line and line[:6] != "MODEL ":
		line = pdbf.readline()
pdbf.close()

coords = numpy.array(coords)
averaged = numpy.mean(coords, axis=2)
stddev = numpy.std(coords, axis=2)

atom_points = vtk.vtkPoints()
atom_points.SetNumberOfPoints(natoms)

tensors = vtk.vtkFloatArray()
tensors.SetNumberOfComponents(9)
tensors.SetNumberOfTuples(natoms)
tensors.SetName("ellipsoids")

for i in range(natoms):
	a = averaged[i]
	atom_points.InsertPoint(i, a[0], a[1], a[2])
	r = stddev[i]*3.0
	tensors.SetTuple(i, (r[0], 0.0,  0.0,
	                     0.0, r[1],  0.0,
	                     0.0,  0.0, r[2]))

grid = vtk.vtkUnstructuredGrid()
grid.SetPoints(atom_points)
grid.GetPointData().SetTensors(tensors)

sphere = vtk.vtkSphereSource()
sphere.SetPhiResolution(16)
sphere.SetThetaResolution(16)

ellipsoids = vtk.vtkTensorGlyph()
ellipsoids.SetInputData(grid)
ellipsoids.SetSourceConnection(sphere.GetOutputPort())

ellMapper = vtk.vtkPolyDataMapper()
ellMapper.SetInputConnection(ellipsoids.GetOutputPort())

ellActor = vtk.vtkActor()
ellActor.SetMapper(ellMapper)
ellActor.GetProperty().SetOpacity(0.3)

molecule = vtk.vtkMolecule()

for i in range(natoms):
	s = symbols[i]
	a = averaged[i]
	molecule.AppendAtom(s, a[0], a[1], a[2])

for i in range(natoms):
	a = averaged[i]
	for j in range(i+1, natoms):
		b = averaged[j]
		d = numpy.sqrt(numpy.sum((a-b)**2))
		if d < 2.0:
			molecule.AppendBond(i, j, 1)

molMapper = vtk.vtkMoleculeMapper()
molMapper.UseLiquoriceStickSettings()
molMapper.SetInputData(molecule)

molActor = vtk.vtkActor()
molActor.SetMapper(molMapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(molActor)
renderer.AddActor(ellActor)

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