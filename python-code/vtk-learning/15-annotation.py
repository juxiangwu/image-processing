#coding:utf-8
import vtk
from sys import argv
import numpy as N
import os

symdict = { 'H' : 1, 'C' : 6, 'O' : 8, 'K' : 19 }

def read_xyz(fnm):

	out = {}

	f = open(fnm)

	l = f.readline()
	n = int(l)

	sym = []
	coord = []

	l = f.readline()
	for i in range(n):
		l = f.readline().split()
		sym.append(l[0])
		coord.append((lambda x: float(x)/0.529177209, l[1:4]))

	return sym, N.array(coord)

curdir,fname = os.path.split(__file__)
model_path = os.path.join(curdir,'models/crown.xyz')
symbols, coordinates = read_xyz(model_path)
nat = len(symbols)

molecule = vtk.vtkMolecule()

for i in range(nat):
	s = symbols[i]
	a = coordinates[i]
	molecule.AppendAtom(symdict[s], a[0], a[1], a[2])

for i in range(nat):
	a = coordinates[i]
	for j in range(i+1, nat):
		b = coordinates[j]
		d = N.sqrt(N.sum((a-b)**2))
		if d < 3.0:
			molecule.AppendBond(i, j, 1)

renderer = vtk.vtkRenderer()

text = []
textmappers = []
textactors = []
for i in range(nat):

	a = coordinates[i]
	s = symbols[i]
	if s == "H": break

	text.append(vtk.vtkVectorText())
	text[-1].SetText(s)

	textmappers.append(vtk.vtkPolyDataMapper())
	textmappers[-1].SetInputConnection(text[-1].GetOutputPort())

	textactors.append(vtk.vtkFollower())
	textactors[-1].SetMapper(textmappers[-1])
	textactors[-1].SetScale(0.8, 0.8, 0.8)
	textactors[-1].AddPosition(a[0]-0.5, a[1]+0.5, a[2]+0.5)
	textactors[-1].SetCamera(renderer.GetActiveCamera())

molMapper = vtk.vtkMoleculeMapper()
molMapper.SetInputData(molecule)

molActor = vtk.vtkActor()
molActor.SetMapper(molMapper)

renderer.AddActor(molActor)
for i in range(len(textactors)):
	renderer.AddActor(textactors[i])
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