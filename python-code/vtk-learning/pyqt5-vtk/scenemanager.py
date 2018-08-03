
import vtk

class SceneManager:

	def __init__(self, vtkWidget=None):
		self.ren = vtk.vtkRenderer()
		self.window = vtkWidget.GetRenderWindow()
		self.window.AddRenderer(self.ren)
		self.iren = self.window.GetInteractor()
		self.iren.Initialize()

		# Create plane
		planeSource = vtk.vtkPlaneSource();
		planeSource.SetCenter(0.0, 0.0, 0.0)
		planeSource.SetNormal(0.0, 0.0, 1.0)
		planeSource.SetResolution(10,10)
		planeSource.Update()

		planeMapper = vtk.vtkPolyDataMapper();
		if(vtk.vtkVersion.GetVTKMajorVersion()<= 5):
			planeMapper.SetInput(planeSource.GetOutput())
		else:
			planeMapper.SetInputData(planeSource.GetOutput())


		planeActor = vtk.vtkActor()
		planeActor.SetMapper(planeMapper)

		self.ren.AddActor(planeActor)

		# Create cone
		coneSource = vtk.vtkConeSource()
		coneSource.SetCenter(0.2,0.0,0.15)
		coneSource.SetHeight(0.3)
		coneSource.SetDirection(0.0,0.0,1.0)
		coneSource.SetAngle(30.0)
		coneSource.SetResolution(36)
		coneSource.Update()

		coneMapper = vtk.vtkPolyDataMapper()
		if(vtk.vtkVersion.GetVTKMajorVersion()<= 5):
			coneMapper.SetInput(coneSource.GetOutput())
		else:
			coneMapper.SetInputData(coneSource.GetOutput())

		coneActor = vtk.vtkActor()
		coneActor.SetMapper(coneMapper)
		coneActor.GetProperty().SetColor(1.0,1.0,0.0)
		self.ren.AddActor(coneActor)

		# Create sphere
		sphereSource = vtk.vtkSphereSource()
		sphereSource.SetCenter(0.3,0.3,0.3)
		sphereSource.SetPhiResolution(10)
		sphereSource.SetThetaResolution(36)
		sphereSource.SetRadius(0.15)
		sphereSource.Update()

		sphereMapper = vtk.vtkPolyDataMapper()
		if(vtk.vtkVersion.GetVTKMajorVersion()<= 5):
 			sphereMapper.SetInput(sphereSource.GetOutput())
		else:
			sphereMapper.SetInputData(sphereSource.GetOutput())

		sphereActor = vtk.vtkActor()
		sphereActor.SetMapper(sphereMapper)
		sphereActor.GetProperty().SetColor(0.0,0.5,1.0)
		self.ren.AddActor(sphereActor)

		# Create orientation axes
		axes = vtk.vtkAxesActor()
		axes.SetShaftTypeToCylinder()

		self.orient = vtk.vtkOrientationMarkerWidget()
		self.orient.SetOrientationMarker( axes )
		self.orient.SetInteractor( self.iren )
		self.orient.SetViewport( 0.0, 0.0, 0.2, 0.2 )
		self.orient.SetEnabled(1)		# Needed to set InteractiveOff
		self.orient.InteractiveOff()
		self.orient.SetEnabled(0)

		self.ren.ResetCamera()

	def SetViewXY(self):
		camera = self.ren.GetActiveCamera()
		camera.SetPosition(0.0,0.0,3.09203)
		camera.SetViewUp(0.0,1.0,0.0)
		camera.SetFocalPoint(0.0,0.0,0.5)
		self.window.Render()

	def SetViewXZ(self):
		camera = self.ren.GetActiveCamera()
		camera.SetPosition(0.0,3.09203,0.5)
		camera.SetViewUp(0.0,0.0,1.0)
		camera.SetFocalPoint(0.0,0.0,0.5)
		self.window.Render()

	def SetViewYZ(self):
		camera = self.ren.GetActiveCamera()
		camera.SetPosition(3.09203,0.0,0.5)
		camera.SetViewUp(0.0,0.0,1.0)
		camera.SetFocalPoint(0.0,0.0,0.5)
		self.window.Render()

	def Snapshot(self):
		wintoim=vtk.vtkWindowToImageFilter()
		self.window.Render()
		wintoim.SetInput(self.window)
		wintoim.Update()

		snapshot = vtk.vtkPNGWriter()
		filenamesnap = "snapshot.png"
		snapshot.SetFileName(filenamesnap)
		snapshot.SetInputConnection(0,wintoim.GetOutputPort())
		snapshot.Write()

	def ToggleVisualizeAxis(self, visible):
		self.orient.SetEnabled(1)		# Needed to set InteractiveOff
		self.orient.InteractiveOff()
		self.orient.SetEnabled(visible)
		self.window.Render()

	def ToggleVisibility(self, visibility):
		# iterate through and set each visibility
	    props = self.ren.GetViewProps()
	    props.InitTraversal()
	    for i in range(props.GetNumberOfItems()):
	        props.GetNextProp().SetVisibility(visibility)

	    self.window.Render()
