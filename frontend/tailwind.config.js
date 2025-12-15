/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'strong-up': '#22c55e',
        'up': '#86efac',
        'strong-down': '#ef4444',
        'down': '#fca5a5',
        'neutral': '#e5e7eb',
        'background': '#0f172a',
        'surface': '#1e293b',
        'surface-light': '#334155',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
