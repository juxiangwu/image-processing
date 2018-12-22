from pyvx import vx

context = vx.CreateContext()
images = [
    vx.CreateImage(context, 640, 480, vx.DF_IMAGE_UYVY),
    vx.CreateImage(context, 640, 480, vx.DF_IMAGE_S16),
    vx.CreateImage(context, 640, 480, vx.DF_IMAGE_U8),
]
graph = vx.CreateGraph(context)
virts = [
    vx.CreateVirtualImage(graph, 0, 0, vx.DF_IMAGE_VIRT),
    vx.CreateVirtualImage(graph, 0, 0, vx.DF_IMAGE_VIRT),
    vx.CreateVirtualImage(graph, 0, 0, vx.DF_IMAGE_VIRT),
    vx.CreateVirtualImage(graph, 0, 0, vx.DF_IMAGE_VIRT),
]
vx.ChannelExtractNode(graph, images[0], vx.CHANNEL_Y, virts[0])
vx.Gaussian3x3Node(graph, virts[0], virts[1])
vx.Sobel3x3Node(graph, virts[1], virts[2], virts[3])
vx.MagnitudeNode(graph, virts[2], virts[3], images[1])
vx.PhaseNode(graph, virts[2], virts[3], images[2])
status = vx.VerifyGraph(graph)

if status == vx.SUCCESS:
    status = vx.ProcessGraph(graph)
else:
    print("Verification failed.")
vx.ReleaseContext(context)