// =============================================================================
// Magna Middleware — Camera Input Provider
// =============================================================================
//! V4L2 camera integration with graceful shutdown support.
//!
//! Only compiled when the `camera` feature is enabled.

#[cfg(feature = "camera")]
mod camera_impl {
    use parking_lot::{Condvar, Mutex};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread::{self, JoinHandle};

    use tracing::{error, info, warn};
    use v4l::buffer::Type;
    use v4l::format::fourcc::FourCC;
    use v4l::io::traits::CaptureStream;
    use v4l::prelude::*;
    use v4l::video::Capture;

    use crate::input::provider::{FrameData, InputProvider};
    use magna_middleware::utils::errors::{MiddlewareError, MiddlewareResult};

    struct SharedState {
        frame: Option<FrameData>,
        updated: bool,
        error: Option<String>,
    }

    pub struct CameraInputProvider {
        device_path: String,
        shared: Arc<(Mutex<SharedState>, Condvar)>,
        stop_flag: Arc<AtomicBool>,
        thread_handle: Option<JoinHandle<()>>,
    }

    impl CameraInputProvider {
        pub fn new(path: &str) -> MiddlewareResult<Self> {
            let shared = Arc::new((
                Mutex::new(SharedState {
                    frame: None,
                    updated: false,
                    error: None,
                }),
                Condvar::new(),
            ));

            let stop_flag = Arc::new(AtomicBool::new(false));
            let shared_clone = shared.clone();
            let stop_clone = stop_flag.clone();
            let path_owned = path.to_string();

            let handle = thread::spawn(move || {
                if let Err(e) = Self::capture_loop(&path_owned, shared_clone.clone(), stop_clone) {
                    error!("Camera {} capture loop failed: {}", path_owned, e);
                    let (lock, cvar) = &*shared_clone;
                    let mut state = lock.lock();
                    state.error = Some(e.to_string());
                    state.updated = true;
                    cvar.notify_all();
                }
            });

            Ok(Self {
                device_path: path.to_string(),
                shared,
                stop_flag,
                thread_handle: Some(handle),
            })
        }

        fn capture_loop(
            path: &str,
            shared: Arc<(Mutex<SharedState>, Condvar)>,
            stop: Arc<AtomicBool>,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let dev = Device::with_path(path)?;

            // Try MJPG format for better performance
            let mut fmt = dev.format()?;
            fmt.fourcc = FourCC::new(b"MJPG");
            if let Err(e) = dev.set_format(&fmt) {
                info!("Could not set MJPG, falling back to default: {}", e);
            }

            let fmt = dev.format()?;
            info!("Camera: {}x{} {}", fmt.width, fmt.height, fmt.fourcc);

            let is_yuyv = fmt.fourcc == FourCC::new(b"YUYV");
            let is_mjpg = fmt.fourcc == FourCC::new(b"MJPG");
            if !is_yuyv && !is_mjpg {
                return Err("Unsupported format. Expected MJPG or YUYV.".into());
            }

            let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 3)?;

            // Check stop flag each iteration — enables graceful shutdown
            while !stop.load(Ordering::Relaxed) {
                let (buf, _meta) = stream.next()?;

                let frame_data = if is_mjpg {
                    match turbojpeg::decompress(buf, turbojpeg::PixelFormat::RGB) {
                        Ok(img) => Some(FrameData {
                            width: img.width as u32,
                            height: img.height as u32,
                            rgb_data: img.pixels,
                        }),
                        Err(e) => {
                            warn!("Failed to decode MJPG frame: {:?}", e);
                            None
                        }
                    }
                } else if is_yuyv {
                    Some(Self::decode_yuyv(buf, fmt.width, fmt.height))
                } else {
                    None
                };

                if let Some(fd) = frame_data {
                    let (lock, cvar) = &*shared;
                    let mut state = lock.lock();
                    state.frame = Some(fd);
                    state.updated = true;
                    cvar.notify_all();
                }
            }

            info!("Camera capture loop stopped gracefully");
            Ok(())
        }

        fn decode_yuyv(buf: &[u8], width: u32, height: u32) -> FrameData {
            use rayon::prelude::*;
            let mut rgb_data = vec![0u8; (width * height * 3) as usize];

            rgb_data
                .par_chunks_exact_mut(6)
                .zip(buf.par_chunks_exact(4))
                .for_each(|(rgb, chunk)| {
                    let y0 = chunk[0] as f32;
                    let u = chunk[1] as f32;
                    let y1 = chunk[2] as f32;
                    let v = chunk[3] as f32;

                    let c = y0 - 16.0_f32;
                    let d = u - 128.0_f32;
                    let e1 = v - 128.0_f32;

                    rgb[0] = (1.164_f32 * c + 1.596_f32 * e1).clamp(0.0_f32, 255.0_f32) as u8;
                    rgb[1] = (1.164_f32 * c - 0.392_f32 * d - 0.813_f32 * e1)
                        .clamp(0.0_f32, 255.0_f32) as u8;
                    rgb[2] = (1.164_f32 * c + 2.017_f32 * d).clamp(0.0_f32, 255.0_f32) as u8;

                    let c1 = y1 - 16.0_f32;
                    rgb[3] = (1.164_f32 * c1 + 1.596_f32 * e1).clamp(0.0_f32, 255.0_f32) as u8;
                    rgb[4] = (1.164_f32 * c1 - 0.392_f32 * d - 0.813_f32 * e1)
                        .clamp(0.0_f32, 255.0_f32) as u8;
                    rgb[5] = (1.164_f32 * c1 + 2.017_f32 * d).clamp(0.0_f32, 255.0_f32) as u8;
                });

            FrameData {
                width,
                height,
                rgb_data,
            }
        }

        /// Gracefully stop the camera capture thread.
        pub fn stop(&mut self) {
            self.stop_flag.store(true, Ordering::Relaxed);
            if let Some(handle) = self.thread_handle.take() {
                let _ = handle.join();
            }
        }
    }

    impl Drop for CameraInputProvider {
        fn drop(&mut self) {
            self.stop();
        }
    }

    impl InputProvider for CameraInputProvider {
        fn source_name(&self) -> &str {
            &self.device_path
        }

        fn next_frame(&mut self) -> MiddlewareResult<Option<FrameData>> {
            let (lock, cvar) = &*self.shared;
            let mut state = lock.lock();

            while !state.updated {
                let res = cvar.wait_for(&mut state, std::time::Duration::from_secs(10));
                if res.timed_out() {
                    return Err(MiddlewareError::InputExhausted(
                        "Camera timeout (10s limit)".to_string(),
                    ));
                }
            }

            if let Some(ref e) = state.error {
                return Err(MiddlewareError::InputExhausted(format!(
                    "Camera error: {}",
                    e
                )));
            }

            state.updated = false;
            Ok(state.frame.clone())
        }

        fn is_streaming(&self) -> bool {
            true
        }
    }
}

// Re-export the camera provider when the feature is enabled
#[cfg(feature = "camera")]
pub use camera_impl::CameraInputProvider;

// Stub when camera feature is not enabled
#[cfg(not(feature = "camera"))]
pub struct CameraInputProvider;

#[cfg(not(feature = "camera"))]
impl CameraInputProvider {
    pub fn new(_path: &str) -> magna_middleware::utils::errors::MiddlewareResult<Self> {
        Err(
            magna_middleware::utils::errors::MiddlewareError::BackendUnavailable(
                "Camera support requires the `camera` feature flag. \
             Build with: cargo build --features camera"
                    .into(),
            ),
        )
    }
}
