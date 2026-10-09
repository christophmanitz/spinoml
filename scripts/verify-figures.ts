#!/usr/bin/env tsx
// verify:figures — assert the pure lineFigureSvg renderer meets the print/export contract

import { lineFigureSvg, OKABE_ITO, type Series } from '../src/figures/lineFigure'
import { spawnSync } from 'node:child_process'

const seriesWithGap: Series[] = [
  {
    label: 'train',
    color: '#ff0000',
    points: [
      { x: 1, y: 0.8 },
      { x: 2, y: 0.6 },
      { x: 3, y: null },
      { x: 4, y: 0.4 },
      { x: 5, y: 0.3 },
    ],
  },
  {
    label: 'val',
    color: '#00ff00',
    points: [
      { x: 1, y: 0.9 },
      { x: 2, y: 0.7 },
      { x: 3, y: 0.65 },
      { x: 4, y: 0.5 },
      { x: 5, y: 0.45 },
    ],
  },
]

const seriesSimple: Series[] = [
  {
    label: 'lr',
    color: '#0000ff',
    points: [
      { x: 1, y: 0.01 },
      { x: 2, y: 0.005 },
      { x: 3, y: 0.001 },
      { x: 4, y: 0.0005 },
      { x: 5, y: 0.0001 },
    ],
  },
]

function assert(condition: boolean, msg: string): void {
  if (!condition) throw new Error(`ASSERT: ${msg}`)
}

function checkPrintTheme(svg: string, name: string, expectedLabels: string[]): void {
  assert(svg.startsWith('<svg'), `${name}: must start with <svg`)
  assert(!svg.includes('var('), `${name}: print theme must not contain CSS var()`)
  assert(!svg.includes('foreignObject'), `${name}: print theme must not contain foreignObject`)
  assert(!svg.includes('preserveAspectRatio="none"'), `${name}: print theme must not have preserveAspectRatio="none"`)
  assert(svg.includes('xmlns="http://www.w3.org/2000/svg"'), `${name}: must have xmlns`)
  assert(svg.includes('<rect x="0" y="0"'), `${name}: must have white background rect`)
  assert(svg.includes('fill="#ffffff"') || svg.includes("fill='#ffffff'"), `${name}: background must be white`)

  // Check legend labels are present
  for (const label of expectedLabels) {
    assert(svg.includes(label), `${name}: legend must contain '${label}'`)
  }

  // Check OKABE_ITO colours are used (at least as many as series)
  for (let i = 0; i < expectedLabels.length; i++) {
    assert(svg.includes(OKABE_ITO[i]), `${name}: must use OKABE_ITO[${i}] (${OKABE_ITO[i]})`)
  }

  // Check font-size 7pt (or 7) for print
  assert(svg.includes('font-size="7"') || svg.includes("font-size='7'"), `${name}: print theme must use 7pt text`)

  // Check font-family Arial/Helvetica
  assert(
    svg.includes('Arial') && svg.includes('Helvetica'),
    `${name}: print theme must use Arial/Helvetica font-family`
  )

  // Validate XML well-formedness via python
  const result = spawnSync('python3', ['-c', 'import xml.dom.minidom; xml.dom.minidom.parseString(open(0).read())'], {
    input: svg,
    encoding: 'utf8',
  })
  assert(result.status === 0, `${name}: SVG must be well-formed XML (python xml.dom.minidom parse failed: ${result.stderr})`)
}

function checkScreenTheme(svg: string, name: string): void {
  assert(svg.startsWith('<svg'), `${name}: must start with <svg`)
  assert(svg.includes('var('), `${name}: screen theme must contain CSS var()`)
  // Screen theme can have preserveAspectRatio="none" or "xMidYMid meet"
  assert(svg.includes('xmlns="http://www.w3.org/2000/svg"'), `${name}: must have xmlns`)
}

function testDeterminism(): void {
  const svg1 = lineFigureSvg(seriesSimple, { theme: 'print', width: 240, height: 170, xLabel: 'epoch', yLabel: 'lr' })
  const svg2 = lineFigureSvg(seriesSimple, { theme: 'print', width: 240, height: 170, xLabel: 'epoch', yLabel: 'lr' })
  assert(svg1 === svg2, 'Determinism: same input must produce identical output')
}

function testPrintThemeWithGap(): void {
  const svg = lineFigureSvg(seriesWithGap, {
    theme: 'print',
    width: 240,
    height: 170,
    xLabel: 'epoch',
    yLabel: 'loss',
    title: 'Loss',
  })
  checkPrintTheme(svg, 'print-with-gap', ['train', 'val'])
}

function testPrintThemeSimple(): void {
  const svg = lineFigureSvg(seriesSimple, {
    theme: 'print',
    width: 240,
    height: 170,
    xLabel: 'epoch',
    yLabel: 'lr',
    title: 'Learning rate',
  })
  checkPrintTheme(svg, 'print-simple', ['lr'])
}

function testScreenTheme(): void {
  const screenSeries: Series[] = [
    {
      label: 'lr',
      color: 'var(--accent)',
      points: [
        { x: 1, y: 0.01 },
        { x: 2, y: 0.005 },
        { x: 3, y: 0.001 },
        { x: 4, y: 0.0005 },
        { x: 5, y: 0.0001 },
      ],
    },
  ]
  const svg = lineFigureSvg(screenSeries, {
    theme: 'screen',
    width: 600,
    height: 220,
    xLabel: 'epoch',
    yLabel: 'lr',
    title: 'Learning rate',
  })
  checkScreenTheme(svg, 'screen')
}

function testDefaults(): void {
  // Test default width/height
  const svg = lineFigureSvg(seriesSimple, { theme: 'print' })
  assert(svg.includes('width="240"'), 'Default width must be 240')
  assert(svg.includes('height="170"'), 'Default height must be 170')
}

function runAll(): void {
  console.log('Running verify-figures tests...')
  testDeterminism()
  console.log('✓ Determinism')
  testPrintThemeWithGap()
  console.log('✓ Print theme with null gap')
  testPrintThemeSimple()
  console.log('✓ Print theme simple')
  testScreenTheme()
  console.log('✓ Screen theme has var()')
  testDefaults()
  console.log('✓ Default dimensions')
  console.log('All verify-figures tests passed!')
}

runAll()