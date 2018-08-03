#coding:utf-8
import vtk
import os
curdir,fname = os.path.split(__file__)
density = vtk.vtkGaussianCubeReader()
density.SetFileName(os.path.join(curdir,"models/crown.den.cube"))
density.Update()

potential = vtk.vtkGaussianCubeReader()
potential.SetFileName(os.path.join(curdir,"models/crownK.pot.cube"))
potential.SetBScale(20)
potential.SetHBScale(15)
potential.Update()

sphere = vtk.vtkSphereSource()
sphere.SetCenter(0, 0, 0)
sphere.SetRadius(1.0)
sphere.SetThetaResolution(16)
sphere.SetPhiResolution(16)

glyph = vtk.vtkGlyph3D()
glyph.SetInputConnection(potential.GetOutputPort())
glyph.SetSourceConnection(sphere.GetOutputPort())
glyph.OrientOn()
glyph.SetColorMode(1)
glyph.SetScaleMode(2)
glyph.SetScaleFactor(1.0)
glyph.ScalingOn()

atomMapper = vtk.vtkPolyDataMapper()
atomMapper.SetInputConnection(glyph.GetOutputPort())

atomActor = vtk.vtkActor()
atomActor.SetMapper(atomMapper)

tube = vtk.vtkTubeFilter()
tube.SetInputConnection(potential.GetOutputPort())
tube.SetNumberOfSides(16)
tube.SetCapping(0)
tube.SetVaryRadius(0)
tube.SetRadius(0.6)
tube.SetRadiusFactor(10)

bondMapper = vtk.vtkPolyDataMapper()
bondMapper.SetInputConnection(tube.GetOutputPort())
bondMapper.UseLookupTableScalarRangeOff()
bondMapper.SetScalarVisibility(1)
bondMapper.SetScalarModeToDefault()

bondActor = vtk.vtkActor()
bondActor.SetMapper(bondMapper)

denGrad = vtk.vtkImageGradient()
denGrad.SetDimensionality(3)
denGrad.SetInputData(density.GetGridOutput())
denGrad.Update()
denGradOut = denGrad.GetOutput()
# Get rid of the original data
denGradOut.GetPointData().RemoveArray(1)

potGrad = vtk.vtkImageGradient()
potGrad.SetDimensionality(3)
potGrad.SetInputData(potential.GetGridOutput())
potGrad.Update()
potGradOut = potGrad.GetOutput()
# Get rid of the original data
potGradOut.GetPointData().RemoveArray(1)

# Calculate scalar product of density and potential gradient
dot = vtk.vtkImageDotProduct()
dot.SetInput1Data(denGradOut)
dot.SetInput2Data(potGradOut)
dot.Update()
dotOut = dot.GetOutput()
dotOut.GetPointData().GetScalars().SetName('dot-product')

# Invert the direction
math = vtk.vtkImageMathematics()
math.SetConstantK(-1.0)
math.SetOperationToMultiplyByK()
math.SetInputData(potGradOut)

# Assign scalars to vectors
attrib = vtk.vtkAssignAttribute()
attrib.SetInputConnection(math.GetOutputPort())
attrib.Assign(vtk.vtkDataSetAttributes.SCALARS,
              vtk.vtkDataSetAttributes.VECTORS,
              vtk.vtkAssignAttribute.POINT_DATA)
attrib.Update()
efield = attrib.GetOutput()
efield.GetPointData().GetVectors().SetName('electric-field')

# Build new data object with electric field (as vectors) and
# dot-product as scalars
data = vtk.vtkImageData()
data.CopyStructure(attrib.GetOutput())
data.GetPointData().AddArray(dot.GetOutput().GetPointData().GetArray(0))
data.GetPointData().SetScalars(dotOut.GetPointData().GetScalars())
data.GetPointData().AddArray(attrib.GetOutput().GetPointData().GetArray(0))
data.GetPointData().SetVectors(efield.GetPointData().GetVectors())

contour = vtk.vtkContourFilter()
contour.SetInputData(density.GetGridOutput())
contour.SetValue(0, 0.02)

probe = vtk.vtkProbeFilter()
probe.SetInputConnection(contour.GetOutputPort())
probe.SetSourceData(data)
probe.Update()

hog = vtk.vtkHedgeHog()
hog.SetInputConnection(probe.GetOutputPort())
hog.SetScaleFactor(30)

srange = probe.GetOutput().GetScalarRange()
lut = vtk.vtkLookupTable()
lut.SetTableRange(srange)

hogMapper = vtk.vtkPolyDataMapper()
hogMapper.SetInputConnection(hog.GetOutputPort())
hogMapper.UseLookupTableScalarRangeOn()
hogMapper.SetLookupTable(lut)

hogActor = vtk.vtkActor()
hogActor.SetMapper(hogMapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(atomActor)
renderer.AddActor(bondActor)
renderer.AddActor(hogActor)
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