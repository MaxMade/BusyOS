use core::{convert::Infallible, ffi::c_void};

use crate::kernel::ticketlock::{DriverTicketlock, Ticketlock};

use driver_macro::module;
use embedded_graphics::{
    Drawable,
    draw_target::DrawTarget,
    geometry::{OriginDimensions, Point, Size},
    mono_font::{MonoFont, MonoTextStyleBuilder, ascii::FONT_8X13},
    pixelcolor::{Rgb888, RgbColor},
    text::{Baseline, Text},
};

use crate::{
    arch::{
        Paging,
        generic::paging::{PhysicalAddress, ReversePaging},
    },
    driver::{
        console::{ConsoleOutput, ConsoleOutputDriver, Consoles},
        module::{Module, ModuleDriver, Modules},
    },
    kernel::{
        arc::Arc,
        bootinfo::BOOTINFO,
        locking::{CanAcquire, DriverLevelID, LockId, PreviousToken},
    },
    mem::page_frames::PageFrames,
    user::errno::Errno,
};

/// The font every character is drawn in.
const FONT: MonoFont<'static> = FONT_8X13;

/// Colour of the characters.
const FOREGROUND: Rgb888 = Rgb888::new(0xc0, 0xc0, 0xc0);

/// Colour behind the characters, and of a cleared screen.
const BACKGROUND: Rgb888 = Rgb888::BLACK;

/// Columns a tab advances to the next multiple of.
const TAB_WIDTH: usize = 8;

#[derive(Debug, Clone, Copy)]
pub enum Format {
    /// 4 Byte pro Pixel: R, G, B, ungenutzt.
    Rgb,
    /// 4 Byte pro Pixel: B, G, R, ungenutzt.
    Bgr,
}

#[derive(Debug, Clone, Copy)]
pub struct Configuration {
    addr: PhysicalAddress<c_void>,
    height: usize,
    width: usize,
    stride: usize,
    format: Format,
}

impl Configuration {
    pub const fn new(
        addr: PhysicalAddress<c_void>,
        height: usize,
        width: usize,
        stride: usize,
        format: Format,
    ) -> Self {
        Self {
            addr,
            height,
            width,
            stride,
            format,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(packed)]
struct Pixel {
    a: u8,
    b: u8,
    c: u8,
    _unused: u8,
}

struct State {
    /// Every pixel of the screen, row by row, each row `stride` pixels long.
    framebuffer: &'static mut [Pixel],

    /// Text column the next character is drawn in.
    column: usize,

    /// Text row the next character is drawn in.
    row: usize,
}

impl State {
    #[inline]
    fn set_pixel(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8, config: &Configuration) {
        if x >= config.width || y >= config.height {
            panic!(
                "Unable to set pixel ({},{}) for resolution {}x{}",
                x, y, config.width, config.height
            );
        }
        let pixel = &mut self.framebuffer[y * config.stride + x];
        match config.format {
            Format::Rgb => {
                pixel.a = r;
                pixel.b = g;
                pixel.c = b;
            }
            Format::Bgr => {
                pixel.a = b;
                pixel.b = g;
                pixel.c = r;
            }
        }
    }

    /// Writes `text` at the cursor, moving the cursor along and scrolling
    /// once it runs off the last row.
    ///
    /// Understands `\n`, `\r` and `\t`. A character the font has no glyph
    /// for is drawn as the font's replacement glyph.
    fn write_str(&mut self, text: &str, config: &Configuration) {
        let (columns, _) = Self::text_size(config);

        for c in text.chars() {
            match c {
                '\n' => self.new_line(config),
                '\r' => self.column = 0,
                '\t' => {
                    let next = (self.column / TAB_WIDTH + 1) * TAB_WIDTH;
                    // Blank out the cells skipped, so that old text does not
                    // show through.
                    while self.column < next.min(columns) {
                        self.put_char(' ', config);
                    }
                    if self.column >= columns {
                        self.new_line(config);
                    }
                }
                c => {
                    self.put_char(c, config);
                    if self.column >= columns {
                        self.new_line(config);
                    }
                }
            }
        }
    }

    /// Draws `c` in the cell at the cursor and moves the cursor one column
    /// on, without wrapping.
    fn put_char(&mut self, c: char, config: &Configuration) {
        let (cell_width, cell_height) = Self::cell_size();
        let position = Point::new(
            (self.column * cell_width) as i32,
            (self.row * cell_height) as i32,
        );

        let mut encoded = [0; 4];
        let style = MonoTextStyleBuilder::new()
            .font(&FONT)
            .text_color(FOREGROUND)
            .background_color(BACKGROUND)
            .build();
        let _ = Text::with_baseline(c.encode_utf8(&mut encoded), position, style, Baseline::Top)
            .draw(&mut Canvas {
                state: self,
                config,
            });

        self.column += 1;
    }

    /// Moves the cursor to the start of the next row, scrolling the screen up
    /// by one row if the cursor is on the last one.
    fn new_line(&mut self, config: &Configuration) {
        let (_, rows) = Self::text_size(config);

        self.column = 0;
        if self.row + 1 < rows {
            self.row += 1;
            return;
        }

        // Move every text row but the first up by one, and blank the last.
        let (_, cell_height) = Self::cell_size();
        let row_pixels = cell_height * config.stride;
        let text_pixels = rows * row_pixels;

        self.framebuffer.copy_within(row_pixels..text_pixels, 0);
        self.fill(text_pixels - row_pixels..text_pixels, BACKGROUND, config);
    }

    /// Paints every pixel of the screen in the background colour and moves
    /// the cursor to the top left corner.
    fn clear(&mut self, config: &Configuration) {
        self.fill(0..config.height * config.stride, BACKGROUND, config);
        self.column = 0;
        self.row = 0;
    }

    /// Paints the pixels in `range`, indices into the framebuffer, in
    /// `color`.
    fn fill(&mut self, range: core::ops::Range<usize>, color: Rgb888, config: &Configuration) {
        let pixel = Pixel::new(color, &config.format);
        self.framebuffer[range].fill(pixel);
    }

    /// Width and height of one character cell, in pixels.
    const fn cell_size() -> (usize, usize) {
        (
            (FONT.character_size.width + FONT.character_spacing) as usize,
            FONT.character_size.height as usize,
        )
    }

    /// Columns and rows of text that fit on the screen.
    fn text_size(config: &Configuration) -> (usize, usize) {
        let (cell_width, cell_height) = Self::cell_size();
        (config.width / cell_width, config.height / cell_height)
    }
}

impl Pixel {
    /// The pixel showing `color` in the framebuffer's `format`.
    fn new(color: Rgb888, format: &Format) -> Self {
        let (r, g, b) = (color.r(), color.g(), color.b());
        match format {
            Format::Rgb => Self {
                a: r,
                b: g,
                c: b,
                _unused: 0,
            },
            Format::Bgr => Self {
                a: b,
                b: g,
                c: r,
                _unused: 0,
            },
        }
    }
}

/// The framebuffer as an `embedded-graphics` draw target, so that text and
/// shapes can be drawn on it.
///
/// Borrows the locked [`State`] together with the [`Configuration`] that
/// [`State::set_pixel`] needs, which lives outside the lock.
struct Canvas<'a> {
    state: &'a mut State,
    config: &'a Configuration,
}

impl OriginDimensions for Canvas<'_> {
    fn size(&self) -> Size {
        Size::new(self.config.width as u32, self.config.height as u32)
    }
}

impl DrawTarget for Canvas<'_> {
    type Color = Rgb888;
    type Error = Infallible;

    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = embedded_graphics::Pixel<Self::Color>>,
    {
        for embedded_graphics::Pixel(point, color) in pixels {
            // Drawing is clipped to the screen rather than panicking in
            // `set_pixel`: a glyph may stick out of a partial cell at the edge.
            let (Ok(x), Ok(y)) = (usize::try_from(point.x), usize::try_from(point.y)) else {
                continue;
            };
            if x >= self.config.width || y >= self.config.height {
                continue;
            }

            self.state
                .set_pixel(x, y, color.r(), color.g(), color.b(), self.config);
        }

        Ok(())
    }
}

pub struct Framebuffer {
    state: DriverTicketlock<State>,
    configuration: Configuration,
}

unsafe impl Send for Framebuffer {}

module! {
    name: "framebuffer",
    priority: 20,
    driver: crate::driver::framebuffer::Framebuffer,
}

impl Module for Framebuffer {
    /// Takes over the framebuffer the bootloader found, blanks it and
    /// registers the driver.
    ///
    /// A machine the bootloader found no framebuffer on is not an error: the
    /// driver returns without registering.
    ///
    /// # Panics
    ///
    /// If the configuration the bootloader handed over is inconsistent, or
    /// the driver cannot be allocated or registered.
    fn init<Token>(token: Token) -> Result<Token, (Errno, Token)>
    where
        Self: Sized,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let mut token = token;

        // SAFETY: the bootloader fills in the boot information before the
        // kernel runs, and nothing writes it afterwards.
        let configuration = unsafe { BOOTINFO.assume_init_ref() }.framebuffer_config;

        if configuration.addr.is_null() || configuration.width == 0 || configuration.height == 0 {
            return Ok(token);
        }
        if configuration.stride < configuration.width {
            panic!(
                "Invalid framebuffer configuration: stride {} below width {}",
                configuration.stride, configuration.width
            );
        }

        // The framebuffer is physical memory like any other, so the direct map
        // covers it, and no mapping of its own is needed.
        //
        // SAFETY: the bootloader reports `height` rows of `stride` pixels at
        // `addr`, which the firmware set aside for the screen and nothing
        // else in the kernel uses. This driver is created once, so the slice
        // is the only reference to that memory.
        let framebuffer = unsafe {
            let virt_addr = Paging::<PageFrames>::phys_to_virt(configuration.addr);
            core::slice::from_raw_parts_mut(
                virt_addr.as_ptr().cast::<Pixel>(),
                configuration.height * configuration.stride,
            )
        };

        let mut state = State {
            framebuffer,
            column: 0,
            row: 0,
        };

        // Wipe whatever the firmware left on the screen.
        state.clear(&configuration);

        let driver = Self {
            state: DriverTicketlock::new(Ticketlock::new(), state),
            configuration,
        };
        let driver = match Arc::try_new(driver, token) {
            Ok((driver, t)) => {
                token = t;
                driver
            }
            Err((error, _)) => {
                panic!(
                    "Unable to allocate driver instance for framebuffer: {}",
                    error
                );
            }
        };

        // Register as module
        match Modules::register(ModuleDriver::Framebuffer(driver.clone()), token) {
            Ok(t) => token = t,
            Err((error, _)) => {
                panic!("Unable to register framebuffer driver as module: {}", error);
            }
        };

        // Register as console
        match Consoles::register(ConsoleOutputDriver::Framebuffer(driver), token) {
            Ok(t) => token = t,
            Err((error, _)) => {
                panic!(
                    "Unable to register framebuffer driver as console: {}",
                    error
                );
            }
        };

        Ok(token)
    }

    fn name(&self) -> &'static str {
        "framebuffer"
    }
}

impl ConsoleOutput for Framebuffer {
    /// Draws `buffer` at the cursor, returning the number of bytes written,
    /// which is all of them.
    ///
    /// # Errors
    ///
    /// [`Errno::EINVAL`] if the screen is too small to hold a single
    /// character.
    fn write<S, Token>(&self, buffer: &S, token: Token) -> Result<(usize, Token), (Errno, Token)>
    where
        S: AsRef<str>,
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (columns, rows) = State::text_size(&self.configuration);
        if columns == 0 || rows == 0 {
            return Err((Errno::EINVAL, token));
        }

        let text = buffer.as_ref();

        let (mut state, token) = self.state.acquire(token);
        state.write_str(text, &self.configuration);
        let token = state.release(token);

        Ok((text.len(), token))
    }

    /// Blanks the screen and moves the cursor to the top left corner.
    fn clear<Token>(&self, token: Token) -> Result<Token, (Errno, Token)>
    where
        Token: CanAcquire<<DriverLevelID as LockId>::Level> + PreviousToken,
    {
        let (mut state, token) = self.state.acquire(token);
        state.clear(&self.configuration);
        let token = state.release(token);

        Ok(token)
    }

    /// Draws `buffer` at the cursor without taking the lock, which the
    /// panicking code may hold.
    unsafe fn emergency_write<S>(&self, buffer: &S)
    where
        S: AsRef<str>,
    {
        let (columns, rows) = State::text_size(&self.configuration);
        if columns == 0 || rows == 0 {
            return;
        }

        // SAFETY: by the trait's contract nothing else touches the state
        // from here on. A write the panic interrupted halfway is abandoned,
        // which at worst leaves the cursor or one glyph off.
        let state = unsafe { &mut *self.state.data_ptr() };

        state.write_str(buffer.as_ref(), &self.configuration);
    }
}
