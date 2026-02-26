import { BrowserRouter, Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import Analyze from "./pages/Analyze";
import GamePage from "./pages/GamePage"; // 새 페이지 임포트
import Header from "./components/Header"; // Header가 Routes 밖에 있어야 모든 페이지에서 보입니다.

export default function App() {
  return (
    <BrowserRouter>
      <Header /> {/* 모든 페이지 공통 헤더 */}
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/analyze" element={<Analyze />} />
        <Route path="/game" element={<GamePage />} />
      </Routes>
    </BrowserRouter>
  );
}