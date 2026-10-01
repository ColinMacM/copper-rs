# Static Plugins

A plugin is a directory that packages reusable parts of a Copper graph: tasks,
bridges, resources and connections, together with the parameters that configure
them and any files they need. An application selects plugins in its
`copperconfig.ron`. Copper expands them while it reads the configuration, so the
runtime macro, the logreader, topology rendering and every other tool see one
ordinary, fully expanded graph.

The plugin set is fixed when the application is built. The plugin versions and
content pins are recorded in the effective configuration that Copper writes into
the unified log.

Plugin support is opt-in. Enable the `plugins` feature of `cu29` (or of
`cu29-runtime` and `cu29-derive`). Without it, a configuration that has a
`plugins` section fails with an error that names the feature, and the plugin
loader is not compiled in.

[`examples/cu_plugin_demo`](../examples/cu_plugin_demo) is a complete
application built from a plugin.

## 1. Anatomy of a plugin

```text
plugins/pid_loop/
  plugin.ron            manifest
  fragments/loop.ron    configuration template
  python/policy.py      asset (optional)
```

`plugin.ron`:

```ron
(
    format: 1,
    id: "cu-pid-loop",
    version: "0.1.0",
    description: "Single-axis PID loop with setpoint and feedback inputs",
    copper: ">=1.3.0-dev",
    rust: [(package: "cu-pid", version: ">=1.3.0-dev")],
    params: {
        "rate_hz": (kind: Int, min: 1, max: 10000, default: 100),
        "gain_p": (kind: Float, default: 1.0),
        "label": (kind: Str),
        "mode": (kind: Enum(["position", "velocity"]), default: "position"),
    },
    fragments: {
        "loop": (path: "fragments/loop.ron", public: ["pid"]),
    },
    assets: ["python/policy.py"],
)
```

| Field | Meaning |
| --- | --- |
| `format` | Manifest format version. Currently `1`. |
| `id` | Plugin name: lowercase letters, digits and `-`, starting with a letter. |
| `version` | Semantic version of the plugin. |
| `description` | One-line summary. |
| `copper` | Semantic-version requirement on the `cu29-runtime` version that reads the plugin. A development version such as `1.3.0-dev` is matched by a requirement that names it, for example `>=1.3.0-dev`. |
| `rust` | Crates that provide the `type:` paths used by the fragments. `cu-plugin check --app` confirms the application depends on them at a matching version. |
| `params` | Typed parameters (section 3). |
| `fragments` | Named configuration templates. `public` lists the node ids, without the instance prefix, that the application may connect to. |
| `assets` | Files the plugin ships. They are covered by the content pin. |

Unknown fields are errors, so a misspelled key is reported where it is written.

## 2. Fragments

A fragment is a RON file with the sections `tasks`, `bridges`, `resources` and
`cnx`. Application-level settings (`logging`, `runtime`, `monitors`, `missions`,
`log_streaming`, `constants`) and nested `includes` or `plugins` belong to the
application and are not accepted in a fragment. Individual nodes may still set
`missions` themselves.

```ron
(
    tasks: [
        (
            id: "{{instance}}_pid",
            type: "cu_pid::PIDTask",
            config: {"kp": {{gain_p}}, "rate_hz": {{rate_hz}}, "label": "{{label}}"},
        ),
    ],
    cnx: [],
)
```

Placeholders use the `{{name}}` form that `includes` use:

- `{{instance}}` is the instance name chosen by the application.
- `{{plugin_dir}}` is the plugin path exactly as the application wrote it. A
  relative path stays relative, and a task that opens a file resolves it the way
  that task resolves any relative path.
- `{{param}}` is the validated value of a declared parameter.

As with `includes`, a string parameter is inserted without quotes, so the
template writes the quotes. Integer, float and boolean parameters are inserted
as RON literals; a `Float` always carries a decimal point or exponent, so it is
read back as a float.

Every `id` a fragment declares in `tasks`, `bridges` or `resources` starts with
`{{instance}}_`. An application can therefore use one plugin several times, and
two plugins can never claim the same id.

## 3. Parameters

Each parameter has a `kind` and optionally a `default`:

| Kind | Accepts | Extra fields |
| --- | --- | --- |
| `Bool` | `true`, `false` | |
| `Int` | integers | `min`, `max` |
| `Float` | finite numbers; an integer literal is converted | `min`, `max` |
| `Str` | text without `"`, `\` or control characters | |
| `Enum([...])` | one of the listed strings | |

A parameter without a default is required. Copper rejects unknown parameters,
missing required parameters, values of the wrong kind and values outside
`min`/`max`, and names the plugin, the instance and the parameter in the message.

## 4. Using a plugin

Add a `plugins` entry to the application configuration:

```ron
plugins: [
    (
        path: "plugins/pid_loop",
        fragment: "loop",
        instance: "elbow",
        params: {"gain_p": 2.5, "label": "elbow"},
        pin: "blake3:<64 hex digits>",
    ),
],
```

| Field | Meaning |
| --- | --- |
| `path` | Plugin directory, relative to the configuration file. |
| `fragment` | Which fragment of the plugin to instantiate. |
| `instance` | Instance name: lowercase letters, digits and `_`, starting with a letter. Unique per application. |
| `params` | Parameter values. |
| `pin` | Content hash of the plugin (section 5). |
| `dev` | `true` accepts the plugin without a `pin` while it is being edited. Use either `pin` or `dev: true`. |
| `when` | Optional `Feature(...)`/`Not`/`All`/`Any` predicate, evaluated like the one on `includes`. |

Connections to a plugin use the ordinary ids:

```ron
cnx: [
    (src: "encoder", dst: "elbow_pid", msg: "cu_pid::Feedback"),
],
```

A plugin's public nodes (`public` in the manifest) accept connections from the
application. All other nodes of an instance are private to its fragment;
connecting to one from outside the fragment is reported as an error.

A public node can be a task or a bridge. The application reaches a bridge's channels as
`<instance>_<bridge>/<channel>`, for example `src: "obs", dst: "vla_link/obs"`.

Copper binds the ports of a node in the order of its connections in the merged configuration,
and a plugin's connections follow the application's. A task that a plugin provides therefore
declares the ports that the application connects first and the ports that its own fragment
connects last, in both its input and its output tuples.

A `plugins` entry in an included file is resolved relative to that file.

## 5. Pins and provenance

The pin is the BLAKE3 hash of the files that define the plugin: `plugin.ron`,
every fragment and every asset, ordered by path. Copper compares it with the
files on disk each time it reads the configuration at build time. A mismatch stops
the build and prints the hash that was computed, so an intentional update is a
one-line change in the application configuration.

The check happens only while the configuration is read at build time. The
application embeds the expanded configuration, so a built application does not
read the plugin directory again, and editing a plugin file after the build has no
effect on it. The macro tracks the plugin files, so editing one rebuilds the
application.

The effective configuration lists every plugin instance with its id, version,
fragment, instance name, resolved parameters, pin and the node ids it owns. Copper
writes it into the unified log when the application starts, and it is part of the
configuration bundled with the application. A replay therefore identifies the exact
plugin content that produced the recorded run.

## 6. Checks performed while reading the configuration

- the manifest parses and its `copper` requirement accepts the running version;
- the named fragment exists and its placeholders are all defined;
- parameters satisfy the manifest;
- the pin matches, or `dev: true` is set;
- every declared id carries the instance prefix;
- no id collides with an id of the application or of another plugin instance;
- asset files listed in the manifest exist;
- instance names are unique;
- connections into private nodes come only from the instance's own fragment.

## 7. Command-line tool

`cu-plugin` operates on plugin directories and configurations. The `just`
recipes wrap it:

```bash
just plugin-new pid_loop                   # scaffold plugins/pid_loop
just plugin-describe plugins/pid_loop      # parameters, fragments, public nodes, pin
just plugin-validate plugins/pid_loop      # render every fragment with default or sample values
just plugin-pin plugins/pid_loop           # the pin for the application's entry
just plugin-expand examples/cu_plugin_demo/copperconfig.ron
just plugin-demo                           # run the example application
just plugin-check                          # lint and test the plugin system
```

| Command | Result |
| --- | --- |
| `new` | Creates a manifest and a fragment to start from. |
| `describe` | Lists parameters, fragments, public nodes, assets and the pin. |
| `check` | Validates the manifest and renders each fragment. Values come from defaults, `--param key=value`, or a sample per kind. Reports parameters no fragment uses. `--app <Cargo.toml>` also confirms the application depends on the crates listed under `rust`. |
| `pin` | Prints the content pin. |
| `expand` | Prints the resolved configuration of an application; `--summary` lists its nodes, connections and plugin instances; `--features a,b` evaluates `when`. |

## 8. Python assets

A plugin can ship Python files as assets. The task that runs them is an ordinary
`cu-python-task` type named in the fragment, with the script path built from
`{{plugin_dir}}`:

```ron
config: {"script": "{{plugin_dir}}/python/policy.py", "mode": "process"},
```

`cu-python-task` resolves a relative script path against the current directory of
the process, so start the application from the directory the plugin `path` is
relative to, or write `path` as an absolute path.

The pin covers the script as it was at build time. Python assets are read from disk
when the application runs, so a script edited after the build runs as edited and is
no longer the content that was pinned. Treat the directory as part of the
deployment, or copy the assets next to the binary and start from there.
