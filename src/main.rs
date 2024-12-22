use anyhow::Result;
use glam::{vec3, Vec3};
use std::{fmt::Write, fs, ops::Range};

trait ColorTrait {
    fn write_color(&self, buffer: &mut String) -> Result<()>;
}

impl ColorTrait for Vec3 {
    fn write_color(&self, buffer: &mut String) -> Result<()> {
        let r = (self.x * 255.999) as u32;
        let g = (self.y * 255.999) as u32;
        let b = (self.z * 255.999) as u32;

        writeln!(buffer, "{r} {g} {b}")?;

        Ok(())
    }
}

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }

    pub fn color(&self, world: &dyn Hittable) -> Vec3 {
        if let Some(hit_record) = world.hit(self, 0.0..f32::INFINITY) {
            return 0.5 * (hit_record.normal + Vec3::splat(1.0));
        }

        let unit_direction = self.direction.normalize_or_zero();
        let a = 0.5 * (unit_direction.y + 1.0);

        vec3(1.0, 1.0, 1.0).lerp(vec3(0.5, 0.7, 1.0), a)
    }
}

pub struct Camera {
    focal_length: f32,
    viewport_height: f32,
    viewport_width: f32,
    image_height: u32,
    image_width: u32,
    center: Vec3,
}

impl Camera {
    pub fn new(image_width: u32, aspect_ratio: f32) -> Self {
        let viewport_height = 2.0;
        let image_height = (image_width as f32 / aspect_ratio) as u32;
        let viewport_width = viewport_height * (image_width as f32 / image_height as f32);

        Self {
            focal_length: 1.0,
            viewport_height,
            viewport_width,
            image_height,
            image_width,
            center: Vec3::ZERO,
        }
    }

    pub fn viewport_u(&self) -> Vec3 {
        vec3(self.viewport_width, 0.0, 0.0)
    }

    pub fn viewport_v(&self) -> Vec3 {
        vec3(0.0, -self.viewport_height, 0.0)
    }

    pub fn viewport_upper_left(&self) -> Vec3 {
        self.center
            - vec3(0.0, 0.0, self.focal_length)
            - self.viewport_u() / 2.0
            - self.viewport_v() / 2.0
    }

    pub fn render<F>(&self, world: &dyn Hittable, mut on_pixel: F)
    where
        F: FnMut(u32, u32, Vec3),
    {
        let pixel_delta_u = self.viewport_u() / self.image_width as f32;
        let pixel_delta_v = self.viewport_v() / self.image_height as f32;

        let viewport_upper_left = self.viewport_upper_left();
        let pixel00_location = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        for j in 0..self.image_height {
            println!("Scanlines remaining: {}", self.image_height - j);
            for i in 0..self.image_width {
                let pixel_center =
                    pixel00_location + (i as f32 * pixel_delta_u) + (j as f32 * pixel_delta_v);
                let ray_direction = pixel_center - self.center;
                let ray = Ray::new(self.center, ray_direction);

                let pixel_color = ray.color(world);
                on_pixel(i, j, pixel_color);
            }
        }
    }

    pub fn render_to_file(&self, world: &dyn Hittable, path: &str) -> Result<()> {
        let mut buffer = format!("P3\n{} {}\n255\n", self.image_width, self.image_height);

        self.render(world, |_i, _j, pixel_color| {
            pixel_color
                .write_color(&mut buffer)
                .expect("Failed to write pixel color to buffer");
        });

        fs::write(path, buffer)?;
        println!("Done");

        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct HitRecord {
    point: Vec3,
    normal: Vec3,
    t: f32,
    front_face: bool,
}

impl HitRecord {
    pub fn new(point: Vec3, t: f32, ray: &Ray, outward_normal: Vec3) -> Self {
        let front_face = ray.direction.dot(outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        };

        Self {
            point,
            t,
            normal,
            front_face,
        }
    }
}

pub trait Hittable {
    fn hit(&self, ray: &Ray, ray_t: Range<f32>) -> Option<HitRecord>;
}

struct Sphere {
    center: Vec3,
    radius: f32,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, ray_t: Range<f32>) -> Option<HitRecord> {
        let oc = self.center - ray.origin;
        let a = ray.direction.length_squared();
        let h = ray.direction.dot(oc);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let squared_discriminant = discriminant.sqrt();

        let mut root = (h - squared_discriminant) / a;
        if !ray_t.surrounds(&root) {
            root = (h + squared_discriminant) / a;
            if !ray_t.surrounds(&root) {
                return None;
            }
        }

        let point = ray.at(root);
        let outward_normal = (point - self.center) / self.radius;
        let hit_record = HitRecord::new(point, root, ray, outward_normal);

        Some(hit_record)
    }
}

impl Hittable for Vec<Box<dyn Hittable>> {
    fn hit(&self, ray: &Ray, ray_t: Range<f32>) -> Option<HitRecord> {
        let mut result = None;
        let mut closest_so_far = ray_t.end;

        for object in self {
            if let Some(hit_record) = object.hit(ray, ray_t.start..closest_so_far) {
                result = Some(hit_record);
                closest_so_far = hit_record.t;
            }
        }

        result
    }
}

trait RangeExt<Idx> {
    fn surrounds(&self, x: &Idx) -> bool;
}

impl<Idx> RangeExt<Idx> for Range<Idx>
where
    Idx: PartialOrd,
{
    fn surrounds(&self, x: &Idx) -> bool {
        self.start < *x && *x < self.end
    }
}

const EMPTY: Range<f32> = f32::INFINITY..-f32::INFINITY;
const UNIVERSE: Range<f32> = -f32::INFINITY..f32::INFINITY;

fn main() {
    let camera = Camera::new(400, 16.0 / 9.0);
    let world: Vec<Box<dyn Hittable>> = vec![
        Box::new(Sphere {
            center: vec3(0.0, 0.0, -1.0),
            radius: 0.5,
        }),
        Box::new(Sphere {
            center: vec3(0.0, -100.5, -1.0),
            radius: 100.0,
        }),
    ];

    camera
        .render_to_file(&world, "out/image.ppm")
        .expect("Failed to write image to file");
}
