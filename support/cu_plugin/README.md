# cu-plugin

`cu-plugin` creates, inspects, checks and pins Copper static plugins. A plugin is a
directory with a `plugin.ron` manifest, configuration fragments and assets; an
application selects plugins in its `copperconfig.ron`. The full format is described
in [`doc/static-plugins.md`](../../doc/static-plugins.md).

```bash
cu-plugin new pid_loop                  # plugins/pid_loop with a manifest and a fragment
cu-plugin describe plugins/pid_loop     # parameters, fragments, public nodes, assets, pin
cu-plugin check plugins/pid_loop        # render every fragment with default or sample values
cu-plugin pin plugins/pid_loop          # the pin for the application's `plugins` entry
cu-plugin expand copperconfig.ron --summary
```

| Command | Result |
| --- | --- |
| `new <name> [--dir D]` | Creates `D/<name>` (default `plugins/<name>`) with a manifest and one fragment. |
| `describe <path>` | Lists parameters with kinds, ranges and defaults, fragments with their public nodes, assets and the content pin. |
| `check <path> [--param k=v] [--app Cargo.toml]` | Validates the manifest and renders each fragment. Values come from defaults, `--param`, or a sample per kind. Reports parameters no fragment uses. With `--app`, confirms the application depends on the crates in the manifest's `rust` list at matching versions. |
| `pin <path>` | Prints the `blake3:` content pin. |
| `expand <config> [--features a,b] [--summary]` | Prints the resolved configuration, or its nodes, connections and plugin instances. `--features` evaluates `when` predicates. |
