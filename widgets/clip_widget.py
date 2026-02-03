from imgui_bundle import imgui, imgui_color_text_edit as edit
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.dict_utils import EasyDict
from widgets.widget import Widget
import numpy as np

def normalize_vec3(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

class ClippingPlaneWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Clipping Plane")
        self.planes = []  # 每个元素是 EasyDict(normal=float3, d=float)
        self._new_plane = EasyDict(nx=1.0, ny=-1.0, nz=1.0, d=-0.1)
        self.vizPlane = False
        self.scale = 1.0
        self.rr_strategy = False
        self.oenD_gs_strategy = False
        self.rr_clipping = False
        
        

    def __call__(self, show=True):
        if not show:
            return

        # Auto-add a plane and enable vizPlane when widget is first shown
        if len(self.planes) == 0:
            self.planes.append(EasyDict(
                normal=[0.327, -0.079, 1.0],  # default normal
                d=-0.161  # default d
            ))
            self.vizPlane = True

        self.viz.args.scale = self.scale
        self.viz.args.rr_strategy = self.rr_strategy
        self.viz.args.oenD_gs_strategy = self.oenD_gs_strategy
        
        label("New Plane")
        imgui.new_line() 
        _, self._new_plane.nx = imgui.slider_float("nx", self._new_plane.nx, -1.0, 1.0)
        _, self._new_plane.ny = imgui.slider_float("ny", self._new_plane.ny, -1.0, 1.0)
        _, self._new_plane.nz = imgui.slider_float("nz", self._new_plane.nz, -1.0, 1.0)
        _, self._new_plane.d = imgui.slider_float("d", self._new_plane.d, -5.0, 5.0)

        normal_raw = np.array([self._new_plane.nx, self._new_plane.ny, self._new_plane.nz])
        normal_unit = normalize_vec3(normal_raw)

        # normalized
        imgui.text(f"Normalized: [{normal_unit[0]:.3f}, {normal_unit[1]:.3f}, {normal_unit[2]:.3f}]")

        if imgui_utils.button("Add Plane", width=self.viz.button_w):
            self.planes.append(EasyDict(
                normal=normal_unit.tolist(),  # save normalized vector
                d=self._new_plane.d
            ))

        imgui.separator()
        for i, plane in enumerate(self.planes):
            label(f"Plane {i}")
            imgui.new_line() 
            _, plane.normal[0] = imgui.slider_float(f"nx##{i}", plane.normal[0], -1.0, 1.0)
            _, plane.normal[1] = imgui.slider_float(f"ny##{i}", plane.normal[1], -1.0, 1.0)
            _, plane.normal[2] = imgui.slider_float(f"nz##{i}", plane.normal[2], -1.0, 1.0)
            _, plane.d = imgui.slider_float(f"d##{i}", plane.d, -5.0, 5.0)

            
            norm_unit = normalize_vec3(np.array(plane.normal))
            imgui.text(f"Normalized: [{norm_unit[0]:.3f}, {norm_unit[1]:.3f}, {norm_unit[2]:.3f}]")

            imgui.same_line()
            if imgui_utils.button(f"Remove##{i}", width=self.viz.button_w):
                self.planes.pop(i)
                break
        imgui.separator()

        # RR Clipping Settings
        label("RR Clipping Settings")
        imgui.separator()

        # RR Clipping checkbox with auto-linkage
        changed, self.rr_clipping = imgui.checkbox("RR Clipping", self.rr_clipping)

        # Auto-enable related settings when RR Clipping is turned on
        if changed and self.rr_clipping:
            self.rr_strategy = True
            self.oenD_gs_strategy = True


        imgui.separator()

        # Visualization and Optimization Settings
        label("vizPlane", self.viz.label_w)
        _, self.vizPlane = imgui.checkbox("##vizPlane", self.vizPlane)

        # RR Strategy checkbox
        label("RR Strategy", self.viz.label_w)
        _, self.rr_strategy = imgui.checkbox("##rr_strategy", self.rr_strategy)

        # 1D Gaussian Strategy checkbox
        label("1D GS Strategy", self.viz.label_w)
        _, self.oenD_gs_strategy = imgui.checkbox("##oenD_gs_strategy", self.oenD_gs_strategy)

        label("box scale", self.viz.label_w)
        _changed, self.scale = imgui.input_float("##box_scale", self.scale)

        # Output parameters to renderer
        self.viz.args.clipping_planes = [
            EasyDict(normal=normalize_vec3(np.array(p.normal)).tolist(), d=p.d)
            for p in self.planes
        ]
        self.viz.args.vizPlane = self.vizPlane
        self.viz.args.scale = self.scale
        self.viz.args.rr_strategy = self.rr_strategy
        self.viz.args.oenD_gs_strategy = self.oenD_gs_strategy
        self.viz.args.rr_clipping = self.rr_clipping