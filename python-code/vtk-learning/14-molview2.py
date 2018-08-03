#coding:utf-8
from sys import argv
import vtk
import numpy
import os
Colors = { 'H' : (1,1,1,1), 'C' : (0,1,0,1), 'O' : (1,0,0,1), 'K' : (0,1,1,1) }
vdW    = { 'H' : 0.7, 'C' : 1.11, 'O' : 1.12, 'K' : 1.5 }

sphere = vtk.vtkSphereSource()
sphere.SetThetaResolution(16)
sphere.SetPhiResolution(16)

lut = vtk.vtkLookupTable()
keys = Colors.keys()
values = vdW.values()
ncolors = len(keys)
lut.SetNumberOfTableValues(200)
lut.SetTableRange(min(values), max(values))
lut.Build()

types = vtk.vtkFloatArray()
coord = vtk.vtkPoints()
curdir,fname = os.path.split(__file__)
model_path = os.path.join(curdir,'models/crown.xyz')
xyz = open(model_path).readlines()
natoms = int(xyz[0])

types.SetNumberOfValues(natoms)
types.SetName("data")

for i in range(natoms):
	tmp = xyz[2+i].split()
	radius = vdW[tmp[0]]
	types.SetValue(i, radius)
	tmp = (tmp[1:4])
	coord.InsertNextPoint(float(tmp[0]), float(tmp[1]), float(tmp[2]))

grid = vtk.vtkUnstructuredGrid()
grid.SetPoints(coord)
grid.GetPointData().AddArray(types)
grid.GetPointData().SetActiveScalars("data")

glyph = vtk.vtkGlyph3D()
glyph.SetInputData(grid)
glyph.SetSourceConnection(sphere.GetOutputPort())

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(glyph.GetOutputPort())
mapper.ScalarVisibilityOn()
mapper.SetColorModeToMapScalars()
mapper.SetLookupTable(lut)
mapper.UseLookupTableScalarRangeOn()

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.ResetCamera()

window = vtk.vtkRenderWindow()
window.SetSize(800,600)
window.AddRenderer(renderer)

istyle = vtk.vtkInteractorStyleSwitch()
istyle.SetCurrentStyleToTrackballCamera()

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
interactor.SetInteractorStyle(istyle)

interactor.Initialize()
interactor.Start()