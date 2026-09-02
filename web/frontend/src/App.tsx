import { useState, useEffect } from 'react'
import { AppLayout } from './components/layout/AppLayout'
import { LoginPage } from './components/auth/LoginPage'
import { RegisterPage } from './components/auth/RegisterPage'
import { useThemeStore } from './stores/themeStore'
import { useAuthStore } from './stores/authStore'
import { fetchHealth } from './lib/api'

type AuthPage = 'login' | 'register'

export default function App() {
  const theme = useThemeStore((s) => s.theme)
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated())
  const [authPage, setAuthPage] = useState<AuthPage>('login')

  // 初始化主题
  useEffect(() => {
    useThemeStore.getState().setTheme(useThemeStore.getState().theme)
  }, [])

  useEffect(() => {
    fetchHealth()
      .then((health) => {
        console.info(
          `[Novare health] redis=${health.redis.status} enabled=${health.redis.enabled} available=${health.redis.available}`,
        )
      })
      .catch((err) => {
        console.warn('[Novare health] unavailable', err)
      })
  }, [])

  if (!isAuthenticated) {
    return authPage === 'login' ? (
      <LoginPage onSwitchToRegister={() => setAuthPage('register')} />
    ) : (
      <RegisterPage onSwitchToLogin={() => setAuthPage('login')} />
    )
  }

  return <AppLayout />
}
