import { Matrix4, PerspectiveCamera, Scene, Vector3, WebGLRenderer } from 'three';
import type { SplatMesh as SplatMeshType, SparkRenderer as SparkRendererType } from '@sparkjsdev/spark';

// Live WebGL view of a trained 3DGS model (Spark). The renderer is loaded lazily so the article page stays light
// until the reader opens the lab.
export interface SplatCamera { name: string; position: number[]; rotation: number[][]; fx: number; fy: number; width: number; height: number }
export interface SplatRig { target: number[]; up: number[]; radius: number; views: SplatCamera[] }

export class SplatView {
  readonly canvas: HTMLCanvasElement;
  private renderer: WebGLRenderer;
  private scene = new Scene();
  private camera = new PerspectiveCamera(50, 16 / 10, 0.01, 200);
  private spark?: SparkRendererType;
  private mesh?: SplatMeshType;
  private frame = 0;
  private disposed = false;
  private resizeObserver: ResizeObserver;
  orbit = 0;
  elevation = 0;
  distance = 1;
  view: SplatCamera | undefined;

  constructor(canvas: HTMLCanvasElement, private rig: SplatRig) {
    this.canvas = canvas;
    this.renderer = new WebGLRenderer({ canvas, antialias: false, alpha: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x0e1c27, 1);
    this.camera.up.set(rig.up[0], rig.up[1], rig.up[2]);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
  }

  async load(url: string, onProgress: (fraction: number) => void) {
    const { SparkRenderer, SplatMesh } = await import('@sparkjsdev/spark');
    if (this.disposed) return;
    this.spark = new SparkRenderer({ renderer: this.renderer, onDirty: () => this.request() });
    this.scene.add(this.spark);
    this.mesh = new SplatMesh({ url, onProgress: event => { if (event.lengthComputable) onProgress(event.loaded / event.total); } });
    this.scene.add(this.mesh);
    await this.mesh.initialized;
    onProgress(1);
    this.request();
  }

  // Snap to one of the real training/test camera poses (3DGS convention: rotation columns are camera axes, +z forward).
  useView(view: SplatCamera | undefined) {
    this.view = view;
    this.request();
  }

  private place() {
    if (this.view) {
      const [x, y, z] = this.view.position;
      const rotation = this.view.rotation;
      const forward = new Vector3(rotation[0][2], rotation[1][2], rotation[2][2]);
      const down = new Vector3(rotation[0][1], rotation[1][1], rotation[2][1]);
      this.camera.position.set(x, y, z);
      this.camera.up.copy(down.multiplyScalar(-1));
      this.camera.lookAt(new Vector3(x, y, z).add(forward));
      this.camera.fov = 2 * Math.atan(this.view.height / (2 * this.view.fy)) * 180 / Math.PI;
    } else {
      const target = new Vector3(...this.rig.target);
      const up = new Vector3(...this.rig.up).normalize();
      const reference = Math.abs(up.y) < 0.9 ? new Vector3(0, 1, 0) : new Vector3(1, 0, 0);
      const side = new Vector3().crossVectors(up, reference).normalize();
      const front = new Vector3().crossVectors(side, up).normalize();
      const azimuth = this.orbit * Math.PI / 180; const pitch = (35 + this.elevation) * Math.PI / 180;
      const radius = this.rig.radius * this.distance;
      const offset = side.clone().multiplyScalar(Math.cos(pitch) * Math.cos(azimuth) * radius)
        .add(front.clone().multiplyScalar(Math.cos(pitch) * Math.sin(azimuth) * radius))
        .add(up.clone().multiplyScalar(Math.sin(pitch) * radius));
      this.camera.position.copy(target).add(offset);
      this.camera.up.copy(up);
      this.camera.lookAt(target);
      this.camera.fov = 50;
    }
    this.camera.updateProjectionMatrix();
  }

  // Project a world point to normalized [0,1] canvas coordinates for overlays.
  project(point: number[]): [number, number] | undefined {
    const projected = new Vector3(...point).applyMatrix4(new Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse));
    if (projected.z > 1) return undefined;
    return [(projected.x + 1) / 2, (1 - projected.y) / 2];
  }

  // Footprint cut-off in standard deviations (Spark default √8). Small values show individual Gaussians.
  setFootprint(stdDev: number) {
    if (this.spark && this.spark.maxStdDev !== stdDev) { this.spark.maxStdDev = stdDev; this.request(); }
  }

  request() {
    if (this.frame || this.disposed) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.place();
      this.renderer.render(this.scene, this.camera);
    });
  }

  private resize() {
    const { clientWidth, clientHeight } = this.canvas;
    if (!clientWidth || !clientHeight) return;
    this.renderer.setSize(clientWidth, clientHeight, false);
    this.camera.aspect = clientWidth / clientHeight;
    this.request();
  }

  dispose() {
    this.disposed = true;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect();
    this.mesh?.dispose();
    this.renderer.dispose();
  }
}
