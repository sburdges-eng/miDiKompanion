# Design System Documentation

## Overview

This document describes the unified design system for iDAWi, covering both the React/TypeScript UI (`iDAWi/`) and the C++ JUCE UI (`src/ui/`).

## Color System

### Base Colors

**Dark Mode (Default)**
- `--ableton-bg`: `#1a1a1a` - Main background
- `--ableton-surface`: `#2a2a2a` - Surface/panel background
- `--ableton-surface-light`: `#333333` - Lighter surface variant
- `--ableton-border`: `#3a3a3a` - Border color
- `--ableton-border-light`: `#4a4a4a` - Lighter border variant

**Light Mode**
- Automatically switches based on `prefers-color-scheme`
- Colors are inverted for light backgrounds

### Accent Colors

- `--ableton-accent`: `#ff5500` - Primary accent (Ableton orange)
- `--ableton-accent-hover`: `#ff6b1a` - Hover state
- `--ableton-accent-active`: `#e64a00` - Active/pressed state
- `--ableton-accent-light`: `#ff8c4d` - Light variant

### Text Colors

- `--ableton-text`: `#ffffff` - Primary text (WCAG AA compliant)
- `--ableton-text-dim`: `#b8b8b8` - Secondary/dimmed text
- `--ableton-text-muted`: `#888888` - Muted text

### Status Colors

- `--ableton-green`: `#22c55e` - Success/positive state
- `--ableton-yellow`: `#fbbf24` - Warning/caution
- `--ableton-red`: `#ef4444` - Error/negative state

### Emotion Colors

- **Grief**: `#4a6fa5` → `#6b8fc5`
- **Joy**: `#ffd700` → `#ffed4e`
- **Anger**: `#dc143c` → `#ff3d5c`
- **Fear**: `#8b008b` → `#b300b3`
- **Love**: `#ff69b4` → `#ff8cc8`

Each emotion has a base color and a light variant for gradients.

### State Colors

- `--state-hover`: `rgba(255, 255, 255, 0.08)` - Hover overlay
- `--state-active`: `rgba(255, 255, 255, 0.12)` - Active overlay
- `--state-disabled`: `rgba(255, 255, 255, 0.05)` - Disabled state
- `--state-focus`: `var(--ableton-accent)` - Focus ring color
- `--state-focus-ring`: `rgba(255, 85, 0, 0.3)` - Focus ring opacity

## Typography

### Font Scale

- `--font-size-xs`: `0.75rem` (12px)
- `--font-size-sm`: `0.875rem` (14px)
- `--font-size-base`: `1rem` (16px)
- `--font-size-lg`: `1.125rem` (18px)
- `--font-size-xl`: `1.25rem` (20px)
- `--font-size-2xl`: `1.5rem` (24px)
- `--font-size-3xl`: `1.875rem` (30px)
- `--font-size-4xl`: `2.25rem` (36px)

### Line Heights

- `--line-height-tight`: `1.25`
- `--line-height-normal`: `1.5`
- `--line-height-relaxed`: `1.75`

### Letter Spacing

- `--letter-spacing-tight`: `-0.025em`
- `--letter-spacing-normal`: `0`
- `--letter-spacing-wide`: `0.025em`
- `--letter-spacing-wider`: `0.05em`

### Font Families

- **Sans-serif**: `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif`
- **Monospace**: `SF Mono, Monaco, Inconsolata, Fira Mono, Droid Sans Mono, Source Code Pro, monospace`

## Spacing System

Based on 4px base unit:

- `--spacing-1`: `0.25rem` (4px)
- `--spacing-2`: `0.5rem` (8px)
- `--spacing-3`: `0.75rem` (12px)
- `--spacing-4`: `1rem` (16px)
- `--spacing-5`: `1.25rem` (20px)
- `--spacing-6`: `1.5rem` (24px)
- `--spacing-8`: `2rem` (32px)
- `--spacing-10`: `2.5rem` (40px)
- `--spacing-12`: `3rem` (48px)
- `--spacing-16`: `4rem` (64px)

## Border Radius

- `--radius-sm`: `0.25rem` (4px)
- `--radius-md`: `0.5rem` (8px)
- `--radius-lg`: `0.75rem` (12px)
- `--radius-xl`: `1rem` (16px)
- `--radius-full`: `9999px`

## Shadows

- `--shadow-sm`: `0 1px 2px 0 rgba(0, 0, 0, 0.3)`
- `--shadow-md`: `0 4px 6px -1px rgba(0, 0, 0, 0.4)`
- `--shadow-lg`: `0 10px 15px -3px rgba(0, 0, 0, 0.5)`
- `--shadow-xl`: `0 20px 25px -5px rgba(0, 0, 0, 0.6)`

## Animation Timings

- `--transition-fast`: `100ms`
- `--transition-base`: `150ms`
- `--transition-slow`: `300ms`
- `--transition-slower`: `500ms`

## Easing Functions

- `--ease-in-out`: `cubic-bezier(0.4, 0, 0.2, 1)`
- `--ease-out`: `cubic-bezier(0, 0, 0.2, 1)`
- `--ease-in`: `cubic-bezier(0.4, 0, 1, 1)`
- `--ease-bounce`: `cubic-bezier(0.68, -0.55, 0.265, 1.55)`

## Responsive Breakpoints

- `xs`: `475px`
- `sm`: `640px`
- `md`: `768px`
- `lg`: `1024px`
- `xl`: `1280px`
- `2xl`: `1536px`

## Component Patterns

### Buttons

**Primary Button** (`.btn-ableton-active`)
- Background: `--ableton-accent`
- Text: White
- Hover: `--ableton-accent-hover`
- Active: `--ableton-accent-active`
- Focus: Ring with `--state-focus-ring`

**Secondary Button** (`.btn-ableton`)
- Background: `--ableton-surface`
- Border: `--ableton-border`
- Hover: `--ableton-surface-light`
- Active: Scale to 0.98

### Input Fields

- Background: `--ableton-bg`
- Border: `--ableton-border`
- Focus: `--ableton-accent` border with ring
- Placeholder: `--ableton-text-muted`

### Tooltips

- Background: `--ableton-surface`
- Border: `--ableton-border`
- Text: `--ableton-text`
- Shadow: `--shadow-md`
- Delay: 300ms

### Toast Notifications

- Success: Green left border
- Error: Red left border
- Info: Accent left border
- Warning: Yellow left border
- Auto-dismiss: 3 seconds (configurable)

## Accessibility

### WCAG Compliance

All colors meet WCAG AA contrast ratios:
- Text on background: 4.5:1 minimum
- Large text: 3:1 minimum
- Interactive elements: 3:1 minimum

### Keyboard Navigation

- Tab order follows visual hierarchy
- Focus indicators visible on all interactive elements
- Skip links for main content
- Keyboard shortcuts documented in UI

### Screen Reader Support

- ARIA labels on all interactive elements
- ARIA live regions for dynamic content
- Proper heading hierarchy
- Role attributes where needed

### Reduced Motion

Respects `prefers-reduced-motion`:
- Animations reduced to 0.01ms
- Transitions minimized
- No auto-playing animations

## JUCE UI Alignment

The C++ JUCE UI (`KellyLookAndFeel`) uses the same color palette:

- `backgroundDark`: `#1a1a1a`
- `surfaceColor`: `#2a2a2a`
- `primaryColor`: `#ff5500`
- `textPrimary`: `#ffffff`
- `textSecondary`: `#b8b8b8`
- `borderColor`: `#3a3a3a`

## Usage Examples

### React/TypeScript

```tsx
// Using Tailwind classes
<button className="btn-ableton-active">Click me</button>

// Using CSS variables
<div style={{ backgroundColor: 'var(--ableton-surface)' }}>Content</div>
```

### C++ JUCE

```cpp
// Using KellyLookAndFeel colors
g.setColour(KellyLookAndFeel::primaryColor);
g.fillRoundedRectangle(bounds, 8.0f);
```

## Best Practices

1. **Consistency**: Always use design tokens, never hardcode colors
2. **Accessibility**: Test with screen readers and keyboard navigation
3. **Performance**: Use CSS transforms for animations
4. **Responsiveness**: Test on multiple screen sizes
5. **Dark Mode**: Ensure all components work in both themes
