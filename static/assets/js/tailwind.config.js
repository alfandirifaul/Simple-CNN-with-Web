tailwind.config = {
    theme: {
        extend: {
            fontFamily: {
                'sans': ['Inter', 'sans-serif'],
            },
            colors: {
                'brand': {
                    'light': '#38bdf8', // light-blue-400
                    'DEFAULT': '#0ea5e9', // sky-500
                    'dark': '#0284c7'  // sky-600
                },
                'slate': {
                    '950': '#0f172a',
                    '900': '#1e293b',
                    '800': '#334155',
                    '700': '#475569',
                    '600': '#64748b',
                    '500': '#94a3b8',
                    '400': '#cbd5e1',
                    '300': '#e2e8f0',
                }
            },
            keyframes: {
                fadeInUp: {
                    'from': { opacity: '0', transform: 'translateY(20px)' },
                    'to': { opacity: '1', transform: 'translateY(0)' },
                },
                aurora: {
                    'from': { backgroundPosition: '0% 50%' },
                    'to': { backgroundPosition: '100% 50%' },
                }
            },
            animation: {
                'fade-in-up': 'fadeInUp 0.6s ease-in-out forwards',
                'aurora-bg': 'aurora 20s ease-in-out infinite alternate',
            }
        }
    }
}