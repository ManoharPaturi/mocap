import { useState, useEffect } from 'react'
import { Coffee, Settings, Activity, Camera, Database, Download, FileText, Box } from 'lucide-react'

function App() {
  const [fps, setFps] = useState(0)
  const [isConnected, setIsConnected] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [metrics, setMetrics] = useState({})

  // Metrics Poller (Fast)
  useEffect(() => {
    if (!isConnected) return
    const fetchMetrics = async () => {
      try {
        const res = await fetch('http://localhost:8000/metrics')
        const data = await res.json()
        setMetrics(data)
      } catch (e) { }
    }
    const interval = setInterval(fetchMetrics, 200)
    return () => clearInterval(interval)
  }, [isConnected])

  // Poller
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/status')
        const data = await res.json()
        setIsConnected(data.status === 'running')
        setIsRecording(data.recording)
      } catch (e) {
        setIsConnected(false)
      }
    }
    const interval = setInterval(checkStatus, 2000)
    checkStatus()
    return () => clearInterval(interval)
  }, [])

  const toggleRecording = async () => {
    try {
      const endpoint = isRecording ? 'stop' : 'start'
      await fetch(`http://localhost:8000/record/${endpoint}`, { method: 'POST' })
      setIsRecording(!isRecording)
    } catch (e) {
      console.error(e)
    }
  }

  const handleDownload = () => {
    window.open('http://localhost:8000/record/export', '_blank')
  }

  const generateReport = async () => {
    try {
      const res = await fetch('http://localhost:8000/report/generate', { method: 'POST' })
      const data = await res.json()
      if (data.status === 'success') {
        alert(`Report generated at:\n${data.path}`)
      } else {
        alert(`Error: ${data.message || data.error}`)
      }
    } catch (e) {
      console.error(e)
      alert("Failed to connect to server")
    }
  }

  const startViz = async () => {
    try {
      const res = await fetch('http://localhost:8000/viz/3d', { method: 'POST' })
      const data = await res.json()
      if (data.status !== 'success') alert(data.message)
    } catch (e) { console.error(e) }
  }

  return (
    <div className="min-h-screen bg-espresso text-cream font-sans selection:bg-mocha selection:text-white">
      {/* Header */}
      <nav className="border-b border-mocha/30 bg-espresso/95 backdrop-blur shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-gradient-to-br from-mocha to-espresso rounded-lg shadow-inner">
              <Coffee className="w-6 h-6 text-cream" />
            </div>
            <span className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-cream to-latte">
              MoCap Studio
            </span>
          </div>

          <div className="flex items-center gap-4 text-sm font-medium text-latte">
            {isRecording && (
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-red-500/20 border border-red-500/30 text-red-200 animate-pulse">
                <div className="w-2 h-2 rounded-full bg-red-500" />
                RECORDING
              </div>
            )}
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-mocha/20 border border-mocha/20">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
              {isConnected ? 'System Online' : 'Connecting...'}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* Left Panel: Video Feed */}
          <div className="lg:col-span-2 space-y-6">
            <div className={`relative group rounded-2xl overflow-hidden border-4 shadow-2xl bg-black aspect-video transition-colors duration-300 ${isRecording ? 'border-red-500/50' : 'border-mocha/30'}`}>
              {isConnected ? (
                <img
                  src="http://localhost:8000/video_feed"
                  alt="Live Feed"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center flex-col gap-4 text-mocha/50">
                  <Camera className="w-16 h-16 opacity-50" />
                  <p>Waiting for Camera Stream...</p>
                </div>
              )}

              {/* Overlay Badge */}
              <div className="absolute top-4 left-4">
                <div className="px-3 py-1 bg-black/60 backdrop-blur rounded-full text-xs font-mono text-green-400 border border-white/10 flex items-center gap-2">
                  <Activity className="w-3 h-3" />
                  LIVE FEED
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel: Controls */}
          <div className="space-y-6">
            <div className="bg-latte/10 rounded-2xl p-6 border border-mocha/20 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-6 text-latte">
                <Settings className="w-5 h-5" />
                <h2 className="font-semibold">Configuration</h2>
              </div>

              <div className="space-y-6">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-mocha">Model Complexity</label>
                  <select disabled className="w-full bg-espresso border border-mocha/40 rounded-lg px-4 py-2.5 text-cream focus:ring-2 focus:ring-latte focus:border-transparent outline-none transition-all cursor-not-allowed opacity-70">
                    <option>Full (Balanced)</option>
                    <option>Lite (Fastest)</option>
                    <option>Heavy (Accurate)</option>
                  </select>
                  <p className="text-xs text-mocha/60">Currently controlled via config.py</p>
                </div>

                <div className="space-y-2">
                  <div className="p-4 rounded-xl bg-gradient-to-br from-mocha/20 to-transparent border border-mocha/10">
                    <h3 className="text-sm font-medium text-latte mb-1">Session Stats</h3>
                    <div className="grid grid-cols-2 gap-4 mt-3">
                      <div>
                        <p className="text-xs text-mocha">Resolution</p>
                        <p className="text-lg font-mono text-cream">1280x720</p>
                      </div>
                      <div>
                        <p className="text-xs text-mocha">Database</p>
                        <p className={`text-lg font-mono ${isRecording ? 'text-red-400 animate-pulse' : 'text-cream'}`}>
                          {isRecording ? 'REC' : 'Idle'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Live Angles */}
                <div className="space-y-2">
                  <div className="p-4 rounded-xl bg-gradient-to-br from-mocha/20 to-transparent border border-mocha/10">
                    <h3 className="text-sm font-medium text-latte mb-2">Live Biometrics (°)</h3>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">

                      <div className="flex justify-between">
                        <span className="text-mocha">L Elbow</span>
                        <span className="font-mono text-cream">{metrics.Angle_Elbow_L || '-'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-mocha">R Elbow</span>
                        <span className="font-mono text-cream">{metrics.Angle_Elbow_R || '-'}</span>
                      </div>

                      <div className="flex justify-between">
                        <span className="text-mocha">L Knee</span>
                        <span className="font-mono text-cream">{metrics.Angle_Knee_L || '-'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-mocha">R Knee</span>
                        <span className="font-mono text-cream">{metrics.Angle_Knee_R || '-'}</span>
                      </div>

                      <div className="flex justify-between">
                        <span className="text-mocha">L Should</span>
                        <span className="font-mono text-cream">{metrics.Angle_Shoulder_L || '-'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-mocha">R Should</span>
                        <span className="font-mono text-cream">{metrics.Angle_Shoulder_R || '-'}</span>
                      </div>

                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <button
                  onClick={toggleRecording}
                  disabled={!isConnected}
                  className={`w-full font-bold py-3 rounded-xl transition-all shadow-lg active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed ${isRecording ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-mocha hover:bg-mocha/80 text-espresso'}`}
                >
                  <div className="flex items-center justify-center gap-2">
                    <Database className="w-5 h-5" />
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                  </div>
                </button>

                <button
                  onClick={handleDownload}
                  disabled={isRecording || !isConnected}
                  className="w-full bg-espresso border border-mocha/30 hover:bg-mocha/10 text-latte font-bold py-3 rounded-xl transition-all shadow-lg active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="flex items-center justify-center gap-2">
                    <Download className="w-5 h-5" />
                    Download CSV
                  </div>
                </button>
              </div>

              <div className="pt-4 border-t border-mocha/20 space-y-3">
                <h3 className="text-sm font-medium text-latte">AI Analysis</h3>
                <button
                  onClick={startViz}
                  className="w-full bg-espresso border border-cyan-500/30 hover:bg-cyan-900/10 text-cyan-200 font-bold py-3 rounded-xl transition-all shadow-lg active:scale-95"
                >
                  <div className="flex items-center justify-center gap-2">
                    <Box className="w-5 h-5" />
                    Visualize 3D
                  </div>
                </button>

                <button
                  onClick={generateReport}
                  className="w-full bg-espresso border border-purple-500/30 hover:bg-purple-900/10 text-purple-200 font-bold py-3 rounded-xl transition-all shadow-lg active:scale-95"
                >
                  <div className="flex items-center justify-center gap-2">
                    <FileText className="w-5 h-5" />
                    Generate Report
                  </div>
                </button>
              </div>
            </div>

            {isConnected && (
              <div className="text-center">
                <p className="text-xs text-mocha/40 font-mono">rtsp://stream/local/001</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
