use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Ident, LitInt, LitStr, Path, Token,
};

/// One `key: value` entry inside `module! { ... }`.
enum Field {
    Name(LitStr),
    Driver(Path),
    Priority(LitInt),
}

impl Parse for Field {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        input.parse::<Token![:]>()?;

        match key.to_string().as_str() {
            "name" => Ok(Field::Name(input.parse()?)),
            "driver" => Ok(Field::Driver(input.parse()?)),
            "priority" => Ok(Field::Priority(input.parse()?)),
            other => Err(syn::Error::new(
                key.span(),
                format!("unknown field `{other}`, expected `name`, `driver`, or `priority`"),
            )),
        }
    }
}

/// The fully parsed and validated contents of a `module! { ... }` invocation.
struct ModuleInput {
    name: LitStr,
    driver: Path,
    priority: u32,
}

impl Parse for ModuleInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let fields = Punctuated::<Field, Token![,]>::parse_terminated(input)?;

        let mut name: Option<LitStr> = None;
        let mut driver: Option<Path> = None;
        let mut priority: Option<LitInt> = None;

        for field in fields {
            match field {
                Field::Name(v) => {
                    if name.is_some() {
                        return Err(syn::Error::new_spanned(&v, "duplicate `name` field"));
                    }
                    name = Some(v);
                }
                Field::Driver(v) => {
                    if driver.is_some() {
                        return Err(syn::Error::new_spanned(&v, "duplicate `driver` field"));
                    }
                    driver = Some(v);
                }
                Field::Priority(v) => {
                    if priority.is_some() {
                        return Err(syn::Error::new_spanned(&v, "duplicate `priority` field"));
                    }
                    priority = Some(v);
                }
            }
        }

        let name =
            name.ok_or_else(|| syn::Error::new(input.span(), "missing required field `name`"))?;
        let driver = driver
            .ok_or_else(|| syn::Error::new(input.span(), "missing required field `driver`"))?;
        let priority_lit = priority
            .ok_or_else(|| syn::Error::new(input.span(), "missing required field `priority`"))?;

        let priority: u32 = priority_lit.base10_parse()?;
        if priority > 999 {
            return Err(syn::Error::new_spanned(
                &priority_lit,
                "priority must fit in three digits (0..=999)",
            ));
        }

        // `name` must be usable as part of a Rust identifier, since it's
        // spliced directly into the generated function name.
        let name_str = name.value();
        let mut chars = name_str.chars();
        let starts_ok = matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_');
        let rest_ok = chars.all(|c| c.is_ascii_alphanumeric() || c == '_');

        if name_str.is_empty() || !starts_ok || !rest_ok {
            return Err(syn::Error::new_spanned(
                &name,
                "`name` must be a valid Rust identifier fragment \
                 (start with a letter or underscore, then letters/digits/underscores)",
            ));
        }

        Ok(ModuleInput {
            name,
            driver,
            priority,
        })
    }
}

/// Registers a driver module for automatic initialization.
///
/// Expands to a `#[no_mangle]` function placed in the `.init.modules.text`
/// link section, named so that its priority is encoded in both the symbol
/// name (for readability in symbol dumps) and — critically — the
/// per-function link section suffix (so the linker script can sort
/// modules by priority at link time; see the crate-level docs for the
/// required linker script fragment).
///
/// ```ignore
/// module! {
///     name: "foo",
///     driver: crate::foo::bar::Baz,
///     priority: 99
/// }
/// ```
#[proc_macro]
pub fn module(input: TokenStream) -> TokenStream {
    let ModuleInput {
        name,
        driver,
        priority,
    } = parse_macro_input!(input as ModuleInput);
    let name_str = name.value();

    // e.g. "099"
    let priority_str = format!("{priority:03}");

    // e.g. __module_init_099_foo
    let fn_ident = format_ident!("__module_init_{}_{}", priority_str, name_str);

    // e.g. __module_init_099_foo_ptr
    let ptr_ident = format_ident!("__module_init_{}_{}_ptr", priority_str, name_str);

    // Per-function input section name, e.g. ".init.modules.text.099_foo".
    // A UNIQUE name per function is required for the linker's
    // SORT_BY_NAME to actually reorder these — see note below.
    let section_name = format!(".init.modules.text.{priority_str}_{name_str}");

    // Per-function input section name, e.g. ".init.modules.rodata.099_foo".
    // A UNIQUE name per function is required for the linker's
    // SORT_BY_NAME to actually reorder these — see note below.
    let section_name_ptr = format!(".init.modules.rodata.{priority_str}_{name_str}");

    let expanded = quote! {
        #[unsafe(no_mangle)]
        #[unsafe(link_section = #section_name)]
        pub fn #fn_ident() {
            use crate::kernel::locking::{CanAcquire,PreviousToken};

            let root = unsafe { crate::kernel::locking::RootToken::forge() };
            let (level, mut token) = crate::kernel::locking::InitLevel::enter(root);

            token = match <#driver as crate::driver::module::Module>::init(token) {
                Ok(token) => token,
                Err((_errno, token)) => {
                    // TODO(@MaxMade): At least try to log the error...
                    token
                }
            };

            level.leave(token);
        }

        #[used]
        #[unsafe(no_mangle)]
        #[unsafe(link_section = #section_name_ptr)]
        pub static #ptr_ident: fn() = #fn_ident;

    };

    expanded.into()
}
