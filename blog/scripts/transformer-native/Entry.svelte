<script>
  import Page from '../../.viz-build/transformer-explainer/src/routes/+page.svelte';
  import Topbar from '../../.viz-build/transformer-explainer/src/components/Topbar.svelte';
  import { isLoaded, isTextbookOpen, predictedColor } from '../../.viz-build/transformer-explainer/src/store';
  import '../../.viz-build/transformer-explainer/src/styles/app.css';
  import '../../.viz-build/transformer-explainer/src/styles/global.scss';
  isLoaded.set(true);
  isTextbookOpen.set(false);
</script>
<div id="app" style={`--min-screen-width:1300px;--min-column-width:22px;--predicted-color:${predictedColor};`}>
  <div class="transformer-toolbar"><Topbar isActive={true} /></div>
  <div class="transformer-canvas"><Page /></div>
</div>
<style>
  .transformer-toolbar { position: relative; z-index: 20; background: white; }
  .transformer-canvas { position: relative; height: 700px; min-width: 1300px; overflow: clip; }
  .transformer-canvas :global(.steps) { grid-template-rows: minmax(0, 1fr); }
  .transformer-canvas :global(.step) { min-height: 0; grid-template-rows: var(--title-height) minmax(0, 1fr); }
  .transformer-canvas :global(.softmax .content) { min-height: 0; height: var(--content-height); max-height: var(--content-height); overflow-y: auto; overflow-x: hidden; overscroll-behavior-y: contain; scrollbar-gutter: stable; }
  .transformer-canvas :global(.prob-dim) { left: 0 !important; right: 0 !important; width: 100% !important; }
  .transformer-canvas :global(.weight-popover) {
    left: 50% !important;
    top: 24px !important;
    transform: translateX(-50%) !important;
    max-width: calc(100% - 48px);
    max-height: calc(100% - 48px);
    overflow: auto;
    overscroll-behavior: contain;
    scrollbar-gutter: stable;
  }
</style>
