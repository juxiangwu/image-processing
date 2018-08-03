#coding:utf-8
from sys import argv
import vtk
import numpy
import os
colors = { 'H' : (1,1,1), 'C' : (0,1,0), 'O' : (1,0,0), 'K' : (0,1,1) }
radii  = { 'H' : 0.7, 'C' : 1.1, 'O' : 1.1, 'K' : 1.5 }
curdir,fname = os.path.split(__file__)
model_path = os.path.join(curdir,'models/crown.xyz')
xyz = open(model_path).readlines()
natoms = int(xyz[0])
symbols = []
coord = []
for a in xyz[2:2+natoms]:
	tmp = a.split()
	symbols.append(tmp[0])
	coord.append((tmp[1:4]))
coord = numpy.array(coord,dtype=numpy.float32)

sphere = vtk.vtkSphereSource()
sphere.SetThetaResolution(16)
sphere.SetPhiResolution(16)

sphere_mapper = vtk.vtkPolyDataMapper()
sphere_mapper.SetInputConnection(sphere.GetOutputPort())

cylinder = vtk.vtkCylinderSource()
cylinder.SetResolution(16)
cylinder.SetRadius(0.2)
cylinder.SetHeight(1.0)

cylinder_mapper = vtk.vtkPolyDataMapper()
cylinder_mapper.SetInputConnection(cylinder.GetOutputPort())

renderer = vtk.vtkRenderer()
atoms = []
# print(coord[0][0])
for i in range(natoms):
    atoms.append(vtk.vtkActor())
    atoms[-1].SetMapper(sphere_mapper)
    atoms[-1].SetPosition(coord[i][0], coord[i][1], coord[i][2])
    atoms[-1].SetScale(radii[symbols[i]])
    c = colors[symbols[i]]
    atoms[-1].GetProperty().SetColor(c[0], c[1], c[2])
    renderer.AddActor(atoms[-1])

bonds = []
zvec = numpy.array([0., 0., 1.])
for i in range(natoms):
	a = coord[i]
	ra = radii[symbols[i]]
	for j in range(i+1, natoms):
		b = coord[j]
		rb = radii[symbols[j]]
		vec = a-b
		d = numpy.sqrt(numpy.sum((vec)**2))
		if d < ra+rb:
			mid = (a+b)/2.0
			angy = numpy.arctan(vec[1]/vec[0])/numpy.pi*180 + 90
			angz = numpy.arctan(vec[2]/numpy.sqrt(numpy.sum(vec[0]**2 + vec[1]**2)))/numpy.pi*180
			rotax = numpy.cross(vec, zvec)
			bonds.append(vtk.vtkActor())
			bonds[-1].SetMapper(cylinder_mapper)
			bonds[-1].SetScale(1,d,1)
			bonds[-1].SetPosition(mid[0], mid[1], mid[2])
			bonds[-1].RotateZ(angy)
			bonds[-1].RotateWXYZ(angz, rotax[0], rotax[1], rotax[2])
			bonds[-1].GetProperty().SetColor(0.5,0.5,0.5)
			renderer.AddActor(bonds[-1])

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