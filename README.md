<div align="center">
  <img src="docs/logo.png" alt="VoiceStudio" width="88" />
  <h1>VoiceStudio</h1>
  <p>
    <a href="https://trendshift.io/repositories/28176?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-28176" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/repositories/28176" alt="VoiceStudio ranking on Trendshift" width="220" height="48" /></a>
  </p>
  <p><strong>The local voice studio: clone, design, dub, dictate, transcribe and narrate in 646 languages.</strong><br />
  Runs on your own computer and phone. No accounts, no API keys, and nothing leaves your device unless you say yes.</p>
  <p>
    <a href="https://voicestudio.sh/?utm_source=github&utm_medium=readme&utm_campaign=project">Website</a> ·
    <a href="https://github.com/debpalash/VoiceStudio/releases/latest">Download</a> ·
    <a href="#get-started">Get started</a> ·
    <a href="#documentation">Docs</a> ·
    <a href="https://discord.gg/bzQavDfVV9">Discord</a> ·
    <a href="docs/readme/README_CN.md">简体中文</a> ·
    <a href="docs/readme/README_JA.md">日本語</a> ·
    <a href="docs/readme/README_KO.md">한국어</a> ·
    <a href="docs/readme/README_RU.md">Русский</a>
  </p>
  <p>
    <a href="https://github.com/debpalash/VoiceStudio/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/debpalash/VoiceStudio/ci.yml?branch=main" alt="CI" /></a>
    <a href="https://github.com/debpalash/VoiceStudio/releases/latest"><img src="https://img.shields.io/github/v/release/debpalash/VoiceStudio" alt="Latest release" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="AGPL-3.0" /></a>
    <a href="https://supermetry.com"><img src="https://app.supermetry.com/px/zfjiv7psp4ed.svg?c=VoiceStudio&style=stats" alt="views" /></a>
  </p>
  <p>
    <sub>Strategic sponsor</sub><br />
    <a href="https://supermetry.com/?utm_source=voicestudio&utm_medium=sponsor&utm_campaign=strategic-sponsor&utm_content=readme-logo"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/sponsors/supermetry-dark.svg" /><img src="docs/sponsors/supermetry.svg" alt="Supermetry" height="48" /></picture></a>
  </p>
</div>

![A tour of the app: voice cloning, voice design, dubbing, and model management](docs/media/electron/voicestudio.gif)

## What you can do

| Create | Produce | Connect |
| :--- | :--- | :--- |
| Clone a voice or design your own | Dub videos with timed speech | Local API & MCP for agents |
| Dictate with a floating widget | Stories, audiobooks & batch jobs | Reusable recipes & watch folders |

Start with **VoiceStudio** (default, powered by k2-fsa/OmniVoice), or pick from dozens of other engines in the Model Manager. See the [features & engine catalog](docs/feature-catalog.md).

<details>
<summary><strong>Screenshots</strong></summary>

<table>
  <tr>
    <td><img src="docs/media/electron/voice-cloning.png" alt="Voice cloning workspace with the bundled demo voice" width="100%" /></td>
    <td><img src="docs/media/electron/dubbing.png" alt="Video dubbing workspace" width="100%" /></td>
  </tr>
  <tr><td align="center">Voice cloning</td><td align="center">Video dubbing</td></tr>
  <tr>
    <td><img src="docs/media/electron/voice-design.png" alt="Describe a voice in the voice design workspace" width="100%" /></td>
    <td><img src="docs/media/electron/models.png" alt="Install and manage local speech models" width="100%" /></td>
  </tr>
  <tr><td align="center">Voice design</td><td align="center">Local models</td></tr>
</table>

</details>

## Platforms

| Platform | Download | Acceleration | Status |
| :--- | :--- | :--- | :--- |
| **macOS** (Apple Silicon) | `.dmg` · [guide](docs/install/macos.md) | Metal (MPS) | Stable |
| **Windows 10/11** (x64) | `.exe` · [guide](docs/install/windows.md) | NVIDIA CUDA · [AMD ROCm (opt-in)](docs/install/windows-rocm.md) · CPU | Stable |
| **Linux** (x64) | `.AppImage` · `.deb` · `snap` · [Nix](docs/install/nix.md) · [guide](docs/install/linux.md) | NVIDIA CUDA · CPU | Stable |
| **Docker** | [`ghcr.io/debpalash/voicestudio`](docs/install/docker.md) | NVIDIA CUDA · AMD ROCm | Stable |
| **Android** 10+ | `.apk` · [guide](docs/mobile.md#install) | On-device, offline | Preview |
| **iPhone / iPad** (iOS 17+) | `.ipa` (sideload) · [guide](docs/mobile.md#install) | On-device, offline | Preview |
| **Windows on ARM** | `.exe` · [guide](docs/install/windows.md) | CPU (x64 backend under emulation) | Experimental |
| **Intel Mac** | `.dmg` · [why](docs/install/macos.md) | App UI only, connects to a remote backend | Limited |

Desktop and phone installers are on the [latest release](https://github.com/debpalash/VoiceStudio/releases/latest); Docker images come from GHCR. No GPU? The app is fully usable on the CPU, just slower. Native GGUF engines can also use Vulkan on AMD, Intel and NVIDIA GPUs; OmniVoice GGUF picks its GPU and model size automatically on Windows and Linux ([OmniVoice](docs/engines/omnivoice-gguf.md#vulkan-acceleration), [CosyVoice](docs/engines/cosyvoice-gguf.md)). Speed by engine and hardware: [performance](docs/performance.md).

## Get started

**macOS / Linux**

```sh
curl -fsSL https://voicestudio.sh/install | sh
```

**Windows** (PowerShell)

```powershell
irm https://voicestudio.sh/install.ps1 | iex
```

Or download an installer from the table above. Then open **Voice cloning**, pick a voice or add a clean reference recording, type your text and generate. The app asks before it downloads any model.

> [!NOTE]
> On macOS, "VoiceStudio is damaged and can't be opened" means Gatekeeper blocked the app, not that the download is broken. See the [Gatekeeper fix](docs/install/macos.md#gatekeeper-quarantine).

<details>
<summary><strong>More install options</strong>: a specific version, current main, uninstall, from source, with a coding agent</summary>

```sh
curl -fsSL https://voicestudio.sh/install | sh -s -- --version X.Y.Z   # a specific release
curl -fsSL https://voicestudio.sh/install | sh -s -- --main            # build current main
curl -fsSL https://voicestudio.sh/install | sh -s -- --uninstall       # remove the app, keep your data
```

If `voicestudio.sh` is unreachable, run the script straight from the repository:
`curl -fsSL https://raw.githubusercontent.com/debpalash/VoiceStudio/main/scripts/install.sh | sh`.
Prerequisites and behavior: [installer guide](docs/install/script.md).

**From source**

```bash
git clone https://github.com/debpalash/VoiceStudio.git
cd VoiceStudio
bun install
bun run setup:api
bun run dev
```

See [Electron setup](electron/README.md).

**With a coding agent** (Claude Code, Codex, Cursor, …), paste:

```text
Install the VoiceStudio app on this device and verify it works, following
https://github.com/debpalash/VoiceStudio/blob/main/docs/install/agent.md
```

Upgrading from 0.5.3 or older (the Tauri app)? [Install the new app separately](docs/electron-migration.md).

</details>

## Documentation

| Need | Start here |
|---|---|
| Setup help | [Troubleshooting](docs/install/troubleshooting.md) · [Model downloads](docs/downloading-models.md) |
| Guides | [Multi-speaker dubbing](docs/dubbing/multi-speaker-guide.md) · [Expressive speech](docs/expressive-speech.md) · [Exporting dubs](docs/dubbing/export.md) · [PDF Voice Reader](docs/pdf-voice-reader.md) |
| Models & quality | [Engine guides](docs/engines/README.md) · [whisper.cpp ASR](docs/engines/whisper-cpp.md) · [Benchmarks](docs/benchmarks.md) |
| Integrations | [Local API](docs/speech-platform.md) · [MCP](docs/mcp.md) · [Examples](examples/README.md) |
| Phone apps | [iOS and Android](docs/mobile.md) |
| Development | [Contributing](.github/CONTRIBUTING.md) ([sign the CLA](.github/CONTRIBUTING.md#contribution-licensing) before your first PR) · [Changelog](CHANGELOG.md) |

Agent skills: `npx skills add debpalash/VoiceStudio` (**voicestudio** for audio work, **voicestudio-maintainer** for the repo).

## Sponsors

VoiceStudio is free and has no paid cloud behind it. Sponsors keep it shipping: they fund maintainer time, test hardware, code-signing and release infrastructure.

<p>
    <a href="https://supermetry.com/?utm_source=voicestudio&utm_medium=sponsor&utm_campaign=strategic-sponsor&utm_content=readme-logo"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/sponsors/supermetry-dark.svg" /><img src="docs/sponsors/supermetry.svg" alt="Supermetry" height="48" /></picture></a>
</p>

**Your logo could be here.** Development, Company and Strategic sponsor slots are open, from $100 a month; Community sponsors ($25) get a named link. Sponsorship buys recognition, not features.

<a href="https://github.com/sponsors/debpalash"><img src="https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ea4aaa?logo=githubsponsors&logoColor=white" alt="Sponsor on GitHub Sponsors" /></a>
[Ko-fi](https://ko-fi.com/debpalash) · [PayPal](https://paypal.me/palashCoder) · [Tiers and benefits](SPONSORS.md) · [Get your logo listed](https://github.com/debpalash/VoiceStudio/issues/new?template=sponsor.yml)

## Team

### Core team

<table>
<tr>
<td align="center" width="200"><a href="https://github.com/debpalash"><img src="https://github.com/debpalash.png?size=160" width="80" height="80" alt="Palash Debnath"></a><br><b>Palash Debnath</b><br><sub>Creator and Core Team</sub><br><sub><a href="https://palash.dev">palash.dev</a> · <a href="https://github.com/debpalash">GitHub</a> · <a href="https://x.com/idebpalash">X</a></sub></td>
<td align="center" width="200"><a href="https://github.com/velixio"><img src="https://github.com/velixio.png?size=160" width="80" height="80" alt="Shimul Debnath"></a><br><b>Shimul Debnath</b><br><sub>Core Team</sub><br><sub><a href="https://shimul.me">shimul.me</a> · <a href="https://github.com/velixio">GitHub</a> · <a href="https://x.com/shimuldn">X</a></sub></td>
</tr>
</table>

### Contributors

<a href="https://github.com/kevin9327"><img src="https://github.com/kevin9327.png?size=96" width="44" height="44" alt="@kevin9327" title="@kevin9327"></a> <a href="https://github.com/jaketame"><img src="https://github.com/jaketame.png?size=96" width="44" height="44" alt="@jaketame" title="@jaketame"></a> <a href="https://github.com/psiberfunk"><img src="https://github.com/psiberfunk.png?size=96" width="44" height="44" alt="@psiberfunk" title="@psiberfunk"></a> <a href="https://github.com/rudycelekli"><img src="https://github.com/rudycelekli.png?size=96" width="44" height="44" alt="@rudycelekli" title="@rudycelekli"></a> <a href="https://github.com/paoloantinori"><img src="https://github.com/paoloantinori.png?size=96" width="44" height="44" alt="@paoloantinori" title="@paoloantinori"></a> <a href="https://github.com/Chang-Jin-Lee"><img src="https://github.com/Chang-Jin-Lee.png?size=96" width="44" height="44" alt="@Chang-Jin-Lee" title="@Chang-Jin-Lee"></a> <a href="https://github.com/swadhinbiswas"><img src="https://github.com/swadhinbiswas.png?size=96" width="44" height="44" alt="@swadhinbiswas" title="@swadhinbiswas"></a> <a href="https://github.com/BichengWang"><img src="https://github.com/BichengWang.png?size=96" width="44" height="44" alt="@BichengWang" title="@BichengWang"></a> <a href="https://github.com/nirmalchatur"><img src="https://github.com/nirmalchatur.png?size=96" width="44" height="44" alt="@nirmalchatur" title="@nirmalchatur"></a> <a href="https://github.com/Shivendra-Coherent"><img src="https://github.com/Shivendra-Coherent.png?size=96" width="44" height="44" alt="@Shivendra-Coherent" title="@Shivendra-Coherent"></a> <a href="https://github.com/marreiradigital"><img src="https://github.com/marreiradigital.png?size=96" width="44" height="44" alt="@marreiradigital" title="@marreiradigital"></a> <a href="https://github.com/nvtoan0201-swe"><img src="https://github.com/nvtoan0201-swe.png?size=96" width="44" height="44" alt="@nvtoan0201-swe" title="@nvtoan0201-swe"></a>

Thank you to everyone who sends code, reports, translations and ideas. [All contributors](https://github.com/debpalash/VoiceStudio/graphs/contributors) · [Contribute](.github/CONTRIBUTING.md)

## License & responsible use

[AGPL-3.0](LICENSE) covers the app, including commercial use under its terms. Model weights, voices and generated output have their own licences, which the Model Manager shows for each model ([inventory](docs/licensing/model-inventory.md)). Clone a voice only with its owner's permission. Details: [LICENSE-NOTICE.md](LICENSE-NOTICE.md).
