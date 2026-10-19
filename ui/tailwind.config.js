/** @type {import('tailwindcss').Config} */
export default {
    // The app toggles html.theme-dark / html.theme-light (see
    // src/app/bootstrap/theme.ts), so `dark:` variants key off that class.
    darkMode: ["class", ".theme-dark"],
    content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
  	extend: {
  		colors: {
  			primary: {
  				DEFAULT: 'hsl(var(--primary))',
  				foreground: 'hsl(var(--primary-foreground))'
  			},
  			// App surface and text tokens from index.css, so new code can write
  			// bg-surface-1 / text-ink-muted instead of bg-[var(--bg-surface-1)].
  			// They are plain colors: for transparency use color-mix(), since
  			// opacity modifiers cannot apply to these hex variables.
  			surface: {
  				'0': 'var(--bg-surface-0)',
  				'1': 'var(--bg-surface-1)',
  				'2': 'var(--bg-surface-2)',
  				'3': 'var(--bg-surface-3)',
  				'4': 'var(--bg-surface-4)'
  			},
  			ink: {
  				DEFAULT: 'var(--text-primary)',
  				secondary: 'var(--text-secondary)',
  				muted: 'var(--text-muted)',
  				subtle: 'var(--text-subtle)'
  			},
  			line: {
  				subtle: 'var(--border-subtle)',
  				DEFAULT: 'var(--border-default)',
  				strong: 'var(--border-strong)'
  			},
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			secondary: {
  				DEFAULT: 'hsl(var(--secondary))',
  				foreground: 'hsl(var(--secondary-foreground))'
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			accent: {
  				DEFAULT: 'hsl(var(--accent))',
  				foreground: 'hsl(var(--accent-foreground))'
  			},
  			destructive: {
  				DEFAULT: 'hsl(var(--destructive))',
  				foreground: 'hsl(var(--destructive-foreground))'
  			},
  			border: 'hsl(var(--border))',
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))',
  			chart: {
  				'1': 'hsl(var(--chart-1))',
  				'2': 'hsl(var(--chart-2))',
  				'3': 'hsl(var(--chart-3))',
  				'4': 'hsl(var(--chart-4))',
  				'5': 'hsl(var(--chart-5))'
  			}
  		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		}
  	}
  },
  plugins: [require("tailwindcss-animate")],
}
