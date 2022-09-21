// Types
import type { GUI } from 'lil-gui'
import type { RootState } from 'store'

// Fonts
import fnt from '/fonts/MSDF/SairaSemiCondensed-SemiBold-msdf.json?url'
import png from '/fonts/MSDF/SairaSemiCondensed-SemiBold.png?url'

// Utils
import * as THREE from 'three'
import { gsap } from 'gsap'
import lerp from 'utils/lerp'
import { scaleValue, randomIntFromInterval } from 'utils/math'
import { Howl } from 'howler'
import breakpoints from 'utils/breakpoints'
import { disablePageScroll, enablePageScroll } from 'scroll-lock'
import { rootNavigate } from 'components/CustomRouter'
import {
  ClickableMesh,
  StateMaterialSet,
  MouseEventManager,
  ThreeMouseEventType
} from '@masatomakino/threejs-interactive-object'
import { MSDFTextGeometry, uniforms } from '../utils/MSDFText'
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader'
import StoreWatcher from '../utils/StoreWatcher'

// Shaders
import vertexShader from './shaders/portfolio/vertex'
import fragmentShader from './shaders/portfolio/fragment'
import captionVertexShader from './shaders/title/vertex'
import captionFragmentShader from './shaders/title/fragment'

// Components
import Experience from '../Experience'

type DebugObject = {
  offsetX: number
  offsetY: number
  offsetZ: number
  iColorOuter: THREE.Color
  iColorInner: THREE.Color
}

type Project = {
  name: string
  url: string
  video: HTMLVideoElement
}

export default class Portfolio {
  experience: Experience
  scene: Experience['scene']
  world: Experience['world']
  resources: Experience['resources']
  debug: Experience['debug']
  time: Experience['time']
  sizes: Experience['sizes']
  debugFolder: GUI | undefined
  group!: THREE.Group
  groupPosition!: THREE.Vector3
  items!: THREE.Mesh[]
  debugObject: DebugObject
  material!: StateMaterialSet
  camera: THREE.PerspectiveCamera
  howls: Howl[]
  manager: MouseEventManager | undefined
  projects: Project[]
  scrollBoundaries: [number, number]
  screenSizes: { screenWidth: number; screenHeight: number }
  xPosition: number
  scrollOffset: number
  isVisible: boolean
  visibleItemIndex: number

  constructor() {
    this.isVisible = false
    this.scrollBoundaries = [0, 0]
    this.xPosition = 0
    this.scrollOffset = 0
    this.visibleItemIndex = -1

    this.experience = new Experience()
    this.resources = this.experience.resources
    this.scene = this.experience.scene
    this.world = this.experience.world
    this.sizes = this.experience.sizes

    this.time = this.experience.time

    this.camera = this.experience.world.cameraOnPath.camera
    if (this.camera && this.experience.renderer.canvas) {
      this.manager = new MouseEventManager(this.scene, this.camera, this.experience.renderer.canvas)
    }

    const bells = 4
    this.howls = []
    for (let i = 1; i <= bells; i++) {
      this.howls.push(
        new Howl({
          src: [`/audio/bell${i}.mp3`],
          volume: 1
        })
      )
    }

    this.projects = [
      {
        name: 'Sketchin',
        url: 'sketchin',
        video: document.getElementById('skReel') as HTMLVideoElement
      },
      {
        name: 'AQuest',
        url: 'aquest',
        video: document.getElementById('aqReel') as HTMLVideoElement
      },
      {
        name: 'Fastweb',
        url: 'fastweb',
        video: document.getElementById('fbReel') as HTMLVideoElement
      },
      {
        name: 'Feudi',
        url: 'feudi',
        video: document.getElementById('feudiReel') as HTMLVideoElement
      },
      {
        name: 'Claraluna',
        url: 'claraluna',
        video: document.getElementById('claralunaReel') as HTMLVideoElement
      }
    ]

    this.debug = this.experience.debug

    if (this.debug.active) {
      this.debugFolder = this.debug.ui?.addFolder('Portfolio').close()
    }

    this.debugObject = {
      offsetX: 1.5,
      offsetY: -0.039,
      offsetZ: -1,
      iColorOuter: new THREE.Color(0x426ff5),
      iColorInner: new THREE.Color(0xc9ddf2)
    }

    this.setItems()
    this.setCaptions()
    this.screenSizes = this.getScreenSizes()

    const storeWatcher = new StoreWatcher()
    storeWatcher.addListener(this.stateChangeHandler.bind(this))
  }

  stateChangeHandler(state: RootState, prevState: RootState) {
    const currentSection = state.section.current
    const prevSection = prevState.section.current

    // Section
    console.log(currentSection, prevSection)
    if (currentSection !== prevSection && currentSection === 'portfolio') {
      this.enterAnimation()
    }

    if (currentSection !== prevSection && prevSection === 'portfolio') {
      this.leaveAnimation()
    }
  }

  setItems() {
    this.group = new THREE.Group()
    this.group.name = 'portfolio'

    // Set the group in front of the camera
    const { offsetX, offsetY, offsetZ } = this.debugObject
    this.group.position.set(offsetX, offsetY, offsetZ)

    this.items = []

    for (let i = 0; i < this.projects.length; i++) {
      // Geometry
      const geometry = new THREE.PlaneGeometry(0.7, 0.5, 12, 12)

      // Play video
      const { video, name, url } = this.projects[i]
      video.play()

      const uniforms = {
        iFactor: { value: 2 },
        iChannel0: { value: new THREE.VideoTexture(video) },
        iChannel1: { value: this.resources.items.noise },
        iColorOuter: { value: this.debugObject.iColorOuter },
        iColorInner: { value: this.debugObject.iColorInner },
        iOffset: { value: new THREE.Vector2(0.01, 0.01) }
      }

      // Material
      const material = new StateMaterialSet({
        normal: new THREE.ShaderMaterial({
          transparent: true,
          uniforms,
          vertexShader,
          fragmentShader
        })
      })

      const clickableMesh: ClickableMesh & { pathname?: string } = new ClickableMesh({
        geo: geometry,
        material
      })
      clickableMesh.castShadow = false
      clickableMesh.receiveShadow = false
      clickableMesh.name = name
      clickableMesh.pathname = url
      clickableMesh.position.set(i * 1.1, 0, 0)

      clickableMesh.addEventListener(ThreeMouseEventType.CLICK, (e) => {
        console.log('clickableMesh click handler. isVisible', this.isVisible)
        if (!this.isVisible) return
        rootNavigate(e.model.view.pathname)
      })
      clickableMesh.addEventListener(ThreeMouseEventType.OVER, (e) => {
        window.store.dispatch.pointer.setType('hover')
        const index = randomIntFromInterval(1, this.howls.length)
        console.log([...this.howls], index)
        this.howls[index].play()
      })
      clickableMesh.addEventListener(ThreeMouseEventType.OUT, () => {
        window.store.dispatch.pointer.setType('default')
      })

      this.group.add(clickableMesh)
      this.items[i] = clickableMesh
    }

    this.setScale()
    this.setBoundaries()

    this.camera.add(this.group)

    // Debug
    if (this.debug.active && this.debugFolder) {
      this.debugFolder
        .add(this.debugObject, 'offsetY')
        .min(-5)
        .max(5)
        .step(0.001)
        .onChange(() => {
          this.group.position.y = this.debugObject.offsetY
        })
      this.debugFolder
        .add(this.debugObject, 'offsetZ')
        .min(-5)
        .max(5)
        .step(0.001)
        .onChange(() => {
          this.group.position.z = this.debugObject.offsetZ
        })
      this.debugFolder.addColor(this.debugObject, 'iColorOuter').onChange((v: string) => {
        for (const item of this.items) {
          // @ts-ignore
          item.material.uniforms.iColorOuter.value = new THREE.Color(v)
        }
      })
      this.debugFolder.addColor(this.debugObject, 'iColorInner').onChange((v: string) => {
        for (const item of this.items) {
          // @ts-ignore
          item.material.uniforms.iColorInner.value = new THREE.Color(v)
        }
      })
    }
  }

  setCaptions() {
    const promises = [this.loadFontAtlas(png), this.loadFont(fnt)]

    return Promise.all(promises).then(([atlas, fnt]) => {
      const font = (fnt as { data: any }).data

      for (let i = 0; i < this.items.length; i++) {
        const geometry = new MSDFTextGeometry({
          text: this.items[i].name,
          font
        })
        const material = new THREE.ShaderMaterial({
          side: THREE.DoubleSide,
          transparent: true,
          defines: {
            IS_SMALL: false
          },
          extensions: {
            derivatives: true
          },
          uniforms: {
            // Common
            ...uniforms.common,

            // Rendering
            ...uniforms.rendering,

            // Strokes
            ...uniforms.strokes,

            uStrokeOutsetWidth: { value: 0.1 },
            uStrokeInsetWidth: { value: 0.0 },
            uProgress1: { value: 1 },
            uProgress2: { value: 0 },
            uProgress3: { value: 0 },
            uProgress4: { value: 0 },
            uProgress5: { value: 0 },
            uDirection: { value: 1 }
          },
          vertexShader: captionVertexShader,
          fragmentShader: captionFragmentShader
        })
        material.uniforms.uMap.value = atlas

        const mesh = new THREE.Mesh(geometry, material)
        mesh.rotation.x = Math.PI

        this.group.add(mesh)
      }
      this.positionCaptions()
    })
  }

  positionCaptions() {
    const captions = this.group.children.filter(
      (mesh) => (mesh as THREE.Mesh).geometry instanceof MSDFTextGeometry
    )
    for (let i = 0; i < captions.length; i++) {
      const mesh = captions[i]
      if (this.sizes.width >= breakpoints.mdL) {
        const scale = 0.004
        mesh.scale.set(scale, scale, scale)
        mesh.position.set(i * 1.1 - 0.1, -0.27, 0.1)
      } else {
        const scale = 0.0015
        mesh.scale.set(scale, scale, scale)
        mesh.position.set(i * 0.5 - 0.1, -0.1, 0.1)
      }
    }
  }

  enterAnimation() {
    console.log('Portfolio enterAnimation')
    this.isVisible = true
    window.addEventListener('scroll', this.scrollHandler)
  }

  revealItem(index: number) {
    if (!this.items[index]) return
    // @ts-ignore
    gsap.to(this.items[index].material.uniforms.iFactor, {
      value: 3.2,
      duration: 3,
      ease: 'power3.out'
    })
  }

  restoreItems() {
    // for (const item of this.items) {
    //   // @ts-ignore
    //   gsap.killTweensOf(item.material.uniforms.iFactor, 'value')
    //   // @ts-ignore
    //   gsap.set(item.material.uniforms.iFactor, { value: 2 })
    // }
  }

  scrollHandler = () => {
    this.xPosition = scaleValue(window.scrollY, this.scrollBoundaries, [
      this.debugObject.offsetX,
      -this.projects.length - 1
    ])
  }

  leaveAnimation() {
    this.isVisible = false
    this.restoreItems()
    window.removeEventListener('scroll', this.scrollHandler)
  }

  openProjectAnimation() {
    // Get the project name and its card
    const location = window.comingLocation
    const projectName = location.pathname.split('/')[1]

    const item = this.items.find(
      (item) => (item as THREE.Mesh & { pathname: string }).pathname === projectName
    )
    if (!item) throw new Error('Project not found')

    // Disable scroll
    requestAnimationFrame(() => {
      disablePageScroll()
    })

    // move object to scene without changing it's world orientation
    // restored in closeProjectAnimation
    this.scene.attach(item)

    this.camera.updateMatrixWorld()

    const distanceFromCamera = 0.5
    const target = new THREE.Vector3(0, 0, -distanceFromCamera)
    target.applyMatrix4(this.camera.matrixWorld)

    // Position
    gsap.to(item.position, {
      x: target.x,
      y: target.y,
      z: target.z,
      duration: 1,
      ease: 'power4.in'
    })

    // Transition
    gsap.to(item.rotation, {
      x: this.camera.rotation.x,
      y: this.camera.rotation.y,
      z: this.camera.rotation.z,
      duration: 1,
      ease: 'power4.in'
    })
  }

  closeProjectAnimation() {
    const name = window.currentLocation.pathname.split('/')[1]

    const index = this.items.findIndex(
      (item) => (item as THREE.Mesh & { pathname: string }).pathname === name
    )
    const item = this.items[index]
    if (!item) throw new Error('Project not found')

    // move object to parent without changing it's world orientation
    this.group.attach(item)

    this.camera.updateMatrixWorld()

    // Position
    gsap.to(item.position, {
      x: index * 1.1,
      y: 0,
      z: 0,
      delay: 0.7,
      ease: 'power4.inOut',
      duration: 1.4
    })

    // Transition
    gsap.to(item.rotation, {
      x: 0,
      y: 0,
      z: 0,
      delay: 0.7,
      ease: 'power4.inOut',
      duration: 1.4,
      onComplete: () => {
        // Enable scroll
        enablePageScroll()
      }
    })
  }

  setScale() {
    for (let i = 0; i < this.items.length; i++) {
      const item = this.items[i]
      if (this.sizes.width >= breakpoints.mdL) {
        item.scale.set(1, 1, 1)
        item.position.set(i * 1.1, 0, 0)
      } else {
        const item = this.items[i]
        item.scale.set(0.4, 0.4, 0.4)
        item.position.set(i * 0.5, 0, 0)
      }
    }
  }

  setBoundaries() {
    const cardContainerEl = document.getElementById('card-container')
    if (cardContainerEl) {
      const start = cardContainerEl.offsetTop + this.sizes.height + this.sizes.height * 0.5 // adding half height to enter the cards after the text
      const end = start + cardContainerEl.clientHeight + this.sizes.height * 0.5
      this.scrollBoundaries = [start, end]
    }
  }

  loadFontAtlas(path: string) {
    const promise = new Promise((resolve) => {
      const loader = new THREE.TextureLoader()
      loader.load(path, resolve)
    })

    return promise
  }

  loadFont(path: string) {
    const promise = new Promise((resolve) => {
      const loader = new FontLoader()
      loader.load(path, resolve)
    })

    return promise
  }

  getScreenSizes(): { screenWidth: number; screenHeight: number } {
    // Find out the width of a rendered portion of the scene
    // https://stackoverflow.com/a/13351534/2150128
    const vFOV = THREE.MathUtils.degToRad(this.camera.fov) // convert vertical fov to radians
    const screenHeight = 2 * Math.tan(vFOV / 2) * Math.abs(this.debugObject.offsetZ) // visible height
    const screenWidth = screenHeight * this.camera.aspect // visible width
    return { screenWidth, screenHeight }
  }

  update() {
    if (!this.isVisible) return
    const scrollY = window.scrollY
    this.scrollOffset = lerp(this.scrollOffset, scrollY, 0.1)
    const offset = (scrollY - this.scrollOffset) * 0.0002
    // Update iOffset

    let i = 0
    for (const item of this.items) {
      ;(item.material as THREE.ShaderMaterial).uniforms.iOffset.value.set(-offset, 0.0)

      const itemX = this.group.position.x + i * (this.sizes.width >= breakpoints.mdL ? 1.1 : 0.5)

      if (itemX < this.screenSizes.screenWidth && this.visibleItemIndex < i) {
        this.visibleItemIndex = i
        this.revealItem(i)
      }

      i++
    }

    gsap.to(this.group.position, {
      x: this.xPosition,
      duration: 2
    })
  }

  resize() {
    this.setScale()
    this.setBoundaries()
    this.positionCaptions()
  }
}
