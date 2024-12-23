use anyhow::Result;
use glam::{vec3, Vec3};
use image::RgbImage;
use rand::Rng;
use std::{fmt::Write, fs, ops::Range};

trait Vec3Ext {
    fn color(&self) -> [u8; 3];
    fn write_color(&self, buffer: &mut String) -> Result<()>;
    fn sample_square() -> Self;
    fn random() -> Self;
    fn random_range(min: f32, max: f32) -> Self;
    fn random_unit_vector() -> Self;
    fn random_on_hemisphere(normal: Self) -> Self;
    fn near_zero(&self) -> bool;
}

impl Vec3Ext for Vec3 {
    fn color(&self) -> [u8; 3] {
        let intensity = 0.0..0.999;

        let r = linear_to_gamma(self.x);
        let g = linear_to_gamma(self.y);
        let b = linear_to_gamma(self.z);

        let r = (intensity.clamp(r) * 256.0) as u8;
        let g = (intensity.clamp(g) * 256.0) as u8;
        let b = (intensity.clamp(b) * 256.0) as u8;

        [r, g, b]
    }

    fn write_color(&self, buffer: &mut String) -> Result<()> {
        let intensity = 0.0..0.999;

        let r = linear_to_gamma(self.x);
        let g = linear_to_gamma(self.y);
        let b = linear_to_gamma(self.z);

        let r = (intensity.clamp(r) * 256.0) as u32;
        let g = (intensity.clamp(g) * 256.0) as u32;
        let b = (intensity.clamp(b) * 256.0) as u32;

        writeln!(buffer, "{r} {g} {b}")?;

        Ok(())
    }

    fn sample_square() -> Self {
        vec3(
            rand::random::<f32>() - 0.5,
            rand::random::<f32>() - 0.5,
            0.0,
        )
    }

    fn random() -> Self {
        vec3(
            rand::random::<f32>(),
            rand::random::<f32>(),
            rand::random::<f32>(),
        )
    }

    fn random_range(min: f32, max: f32) -> Self {
        let mut rng = rand::thread_rng();
        vec3(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        )
    }

    fn random_unit_vector() -> Self {
        loop {
            let p = Vec3::random_range(-1.0, 1.0);
            let length_squared = p.length_squared();
            if 1e-160 < length_squared && length_squared <= 1.0 {
                return p / length_squared.sqrt();
            }
        }
    }

    fn random_on_hemisphere(normal: Self) -> Self {
        let on_unit_sphere = Vec3::random_unit_vector();
        if on_unit_sphere.dot(normal) > 0.0 {
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
    }

    fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }

    // Looks like glam already implements these, only keeping them for reference
    //fn reflect(&self, normal: Self) -> Self {
    //    self - 2.0 * self.dot(normal) * normal
    //}
    //
    //fn refract(&self, normal: Self, eta_i_over_eta_t: f32) -> Self {
    //    let cos_theta = f32::min(-self.dot(normal), 1.0);
    //
    //    let r_out_perpendicular = eta_i_over_eta_t * (self + cos_theta * normal);
    //    let r_out_parallel = -((1.0 - r_out_perpendicular.length_squared()).abs().sqrt()) * normal;
    //
    //    r_out_perpendicular + r_out_parallel
    //}
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

    pub fn color(&self, depth: u32, world: &dyn Hittable) -> Vec3 {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if depth == 0 {
            return Vec3::ZERO;
        }

        if let Some(hit_record) = world.hit(self, 0.001..f32::INFINITY) {
            match hit_record.material_hit.scatter(self, &hit_record) {
                Some((attenuation, scattered_ray)) => {
                    return attenuation * scattered_ray.color(depth - 1, world);
                }
                None => return Vec3::ZERO,
            }
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
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    viewport_upper_left: Vec3,
    pixel00_location: Vec3,
    samples_per_pixel: u32,
    pixel_samples_scale: f32,
    max_depth: u32,
}

impl Camera {
    pub fn new(image_width: u32, aspect_ratio: f32) -> Self {
        let center = Vec3::ZERO;
        let viewport_height = 2.0;
        let focal_length = 1.0;

        let image_height = (image_width as f32 / aspect_ratio) as u32;
        let viewport_width = viewport_height * (image_width as f32 / image_height as f32);

        let viewport_u = vec3(viewport_width, 0.0, 0.0);
        let viewport_v = vec3(0.0, -viewport_height, 0.0);

        let pixel_delta_u = viewport_u / image_width as f32;
        let pixel_delta_v = viewport_v / image_height as f32;

        let viewport_upper_left =
            center - vec3(0.0, 0.0, focal_length) - viewport_u / 2.0 - viewport_v / 2.0;

        let pixel00_location = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        let samples_per_pixel = 500;
        let pixel_samples_scale = 1.0 / samples_per_pixel as f32;

        let max_depth = 50;

        Self {
            focal_length,
            viewport_height,
            viewport_width,
            image_height,
            image_width,
            center,
            pixel_delta_u,
            pixel_delta_v,
            viewport_upper_left,
            pixel00_location,
            samples_per_pixel,
            pixel_samples_scale,
            max_depth,
        }
    }

    pub fn render<F>(&self, world: &dyn Hittable, mut on_pixel: F)
    where
        F: FnMut(u32, u32, Vec3),
    {
        for j in 0..self.image_height {
            println!("Scanlines remaining: {}", self.image_height - j);
            for i in 0..self.image_width {
                let mut pixel_color = Vec3::ZERO;
                for _sample in 0..self.samples_per_pixel {
                    let ray = self.get_ray(i, j);
                    pixel_color += ray.color(self.max_depth, world);
                }

                on_pixel(i, j, self.pixel_samples_scale * pixel_color);
            }
        }
    }

    fn get_ray(&self, i: u32, j: u32) -> Ray {
        let offset = Vec3::sample_square();
        let pixel_sample = self.pixel00_location
            + ((i as f32 + offset.x) * self.pixel_delta_u)
            + ((j as f32 + offset.y) * self.pixel_delta_v);

        let ray_direction = pixel_sample - self.center;

        Ray {
            origin: self.center,
            direction: ray_direction,
        }
    }

    pub fn render_to_ppm_file(&self, world: &dyn Hittable, path: &str) -> Result<()> {
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

    pub fn render_to_png_file(&self, world: &dyn Hittable, path: &str) -> Result<()> {
        let mut image = RgbImage::new(self.image_width, self.image_height);

        self.render(world, |i, j, pixel_color| {
            *image.get_pixel_mut(i, j) = pixel_color.color().into();
        });

        image.save(path)?;
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
    material_hit: Material,
}

impl HitRecord {
    pub fn new(point: Vec3, t: f32, ray: &Ray, outward_normal: Vec3, material: Material) -> Self {
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
            material_hit: material,
        }
    }
}

pub trait Hittable {
    fn hit(&self, ray: &Ray, ray_t: Range<f32>) -> Option<HitRecord>;
}

struct Sphere {
    material: Material,
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
        if !ray_t.surrounds(root) {
            root = (h + squared_discriminant) / a;
            if !ray_t.surrounds(root) {
                return None;
            }
        }

        let point = ray.at(root);
        let outward_normal = (point - self.center) / self.radius;
        let hit_record = HitRecord::new(point, root, ray, outward_normal, self.material);

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
    fn surrounds(&self, x: Idx) -> bool;

    fn clamp(&self, x: Idx) -> Idx;
}

impl<Idx> RangeExt<Idx> for Range<Idx>
where
    Idx: PartialOrd + Copy,
{
    fn surrounds(&self, x: Idx) -> bool {
        self.start < x && x < self.end
    }

    fn clamp(&self, x: Idx) -> Idx {
        if x < self.start {
            self.start
        } else if x > self.end {
            self.end
        } else {
            x
        }
    }
}

const EMPTY: Range<f32> = f32::INFINITY..-f32::INFINITY;
const UNIVERSE: Range<f32> = -f32::INFINITY..f32::INFINITY;

fn linear_to_gamma(linear_component: f32) -> f32 {
    if linear_component > 0.0 {
        linear_component.sqrt()
    } else {
        0.0
    }
}

#[derive(Clone, Copy)]
pub enum Material {
    Lambertian {
        albedo: Vec3,
    },
    Metal {
        albedo: Vec3,
        fuzz: f32,
    },
    Dielectric {
        /// Refractive index in vacuum or air, or the ratio of the material's refractive index over
        /// the refractive index of the enclosing medium.
        refraction_index: f32,
    },
}

impl Material {
    fn scatter(&self, ray_in: &Ray, hit: &HitRecord) -> Option<(Vec3, Ray)> {
        match self {
            Material::Lambertian { albedo } => {
                let mut scatter_direction = hit.normal + Vec3::random_unit_vector();

                // Catch degenerate scatter direction
                if scatter_direction.near_zero() {
                    scatter_direction = hit.normal;
                }

                let scattered = Ray::new(hit.point, scatter_direction);
                Some((*albedo, scattered))
            }
            Material::Metal { albedo, fuzz } => {
                let mut reflected = ray_in.direction.reflect(hit.normal);
                reflected = reflected.normalize_or_zero() + (fuzz * Vec3::random_unit_vector());

                let scattered = Ray::new(hit.point, reflected);

                if scattered.direction.dot(hit.normal) > 0.0 {
                    Some((*albedo, scattered))
                } else {
                    // We scattered below the surface, so the ray is absorbed.
                    None
                }
            }
            Material::Dielectric { refraction_index } => {
                let attenuation = Vec3::ONE;
                let refraction_index = if hit.front_face {
                    1.0 / refraction_index
                } else {
                    *refraction_index
                };

                let unit_direction = ray_in.direction.normalize_or_zero();
                let cos_theta = f32::min(-unit_direction.dot(hit.normal), 1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let cannot_refract = refraction_index * sin_theta > 1.0;
                let direction = if cannot_refract
                    || reflectance(cos_theta, refraction_index) > rand::random()
                {
                    unit_direction.reflect(hit.normal)
                } else {
                    unit_direction.refract(hit.normal, refraction_index)
                };

                let scattered = Ray::new(hit.point, direction);

                Some((attenuation, scattered))
            }
        }
    }
}

// Schlick's approximation for reflectance.
fn reflectance(cosine: f32, refraction_index: f32) -> f32 {
    let mut r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 *= r0;

    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

fn main() {
    let camera = Camera::new(3840, 16.0 / 9.0);
    let world: Vec<Box<dyn Hittable>> = vec![
        // Ground
        Box::new(Sphere {
            center: vec3(0.0, -100.5, -1.0),
            radius: 100.0,
            material: Material::Lambertian {
                albedo: vec3(0.8, 0.8, 0.0),
            },
        }),
        // Center
        Box::new(Sphere {
            center: vec3(0.0, 0.0, -1.2),
            radius: 0.5,
            material: Material::Lambertian {
                albedo: vec3(0.1, 0.2, 0.5),
            },
        }),
        // Left
        Box::new(Sphere {
            center: vec3(-1.0, 0.0, -1.0),
            radius: 0.5,
            material: Material::Dielectric {
                refraction_index: 1.5,
            },
        }),
        // Bubble
        Box::new(Sphere {
            center: vec3(-1.0, 0.0, -1.0),
            radius: 0.4,
            material: Material::Dielectric {
                refraction_index: 1.0 / 1.5, // Air bubble in glass
            },
        }),
        // Right
        Box::new(Sphere {
            center: vec3(1.0, 0.0, -1.0),
            radius: 0.5,
            material: Material::Metal {
                albedo: vec3(1.0, 1.0, 1.0),
                fuzz: 0.0,
            },
        }),
    ];

    camera
        .render_to_png_file(&world, "out/image.png")
        .expect("Failed to write image to file");
}
