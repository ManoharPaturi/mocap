/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        espresso: '#4B3621',
        latte: '#D6C0B3',
        cream: '#F5F5DC',
        mocha: '#967969',
      }
    },
  },
  plugins: [],
}
