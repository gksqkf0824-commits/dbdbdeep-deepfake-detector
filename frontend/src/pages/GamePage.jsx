import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

// 이미지 임포트 (경로는 기존 프로젝트 설정을 유지합니다)
import fake01 from '../game_image/fake_01.png';
import fake02 from '../game_image/fake_02.png';
import fake03 from '../game_image/fake_03.png';
import fake04 from '../game_image/fake_04.png';
import fake05 from '../game_image/fake_05.png';
import real01 from '../game_image/real_01.png';
import real02 from '../game_image/real_02.png';
import real03 from '../game_image/real_03.png';
import real04 from '../game_image/real_04.png';
import real05 from '../game_image/real_05.png';

// 원본 데이터 배열
const initialGameData = [
  { id: 1, img: fake01, label: 'fake', aiScore: 18 },
  { id: 2, img: fake02, label: 'fake', aiScore: 37 },
  { id: 3, img: fake03, label: 'fake', aiScore: 66 },
  { id: 4, img: fake04, label: 'fake', aiScore: 47 },
  { id: 5, img: fake05, label: 'fake', aiScore: 41 },
  { id: 6, img: real01, label: 'real', aiScore: 85 },
  { id: 7, img: real02, label: 'real', aiScore: 77 },
  { id: 8, img: real03, label: 'real', aiScore: 18 },
  { id: 9, img: real04, label: 'real', aiScore: 85 },
  { id: 10, img: real05, label: 'real', aiScore: 81 },
];

export default function GamePage() {
  const navigate = useNavigate();
  
  const [gameData, setGameData] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [userAnswers, setUserAnswers] = useState([]);
  const [showResult, setShowResult] = useState(false);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const shuffled = [...initialGameData].sort(() => Math.random() - 0.5);
    setGameData(shuffled);
    setIsReady(true);
  }, []);

  const handleAnswer = (answer) => {
    const newAnswers = [...userAnswers, answer];
    setUserAnswers(newAnswers);
    if (currentIdx < gameData.length - 1) {
      setCurrentIdx(currentIdx + 1);
    } else {
      setShowResult(true);
    }
  };

  const calculateResults = () => {
    let userCorrect = 0;
    let aiCorrect = 0;
    const details = gameData.map((item, idx) => {
      const userPick = userAnswers[idx];
      const aiVerdict = item.aiScore >= 50 ? 'real' : 'fake';
      const userIsRight = userPick === item.label;
      const aiIsRight = aiVerdict === item.label;
      if (userIsRight) userCorrect++;
      if (aiIsRight) aiCorrect++;
      return { ...item, userPick, aiVerdict, userIsRight, aiIsRight };
    });
    return { userCorrect, aiCorrect, details };
  };

  if (!isReady) return null;

  const results = showResult ? calculateResults() : null;

  return (
    <div className="min-h-screen bg-slate-50 pt-32 pb-20 px-6">
      <div className="max-w-5xl mx-auto">
        {!showResult ? (
          /* 게임 진행 화면 */
          <div className="bg-white rounded-[40px] shadow-sm border border-slate-100 p-10 md:p-16 text-center">
            <div className="flex justify-between items-center mb-10">
              <span className="text-sm font-black text-blue-600 bg-blue-50 px-4 py-2 rounded-xl">
                QUESTION {currentIdx + 1}/{gameData.length}
              </span>
              <h1 className="text-2xl font-black text-slate-900 tracking-tight">인간 vs AI: 딥페이크 판별</h1>
            </div>
            <div className="w-full aspect-video bg-slate-50 rounded-[32px] mb-12 overflow-hidden border border-slate-100 shadow-inner flex items-center justify-center">
              <img 
                src={gameData[currentIdx].img} 
                alt="Quiz" 
                className="max-h-full object-contain pointer-events-none" 
              />
            </div>
            <div className="grid grid-cols-2 gap-6">
              <button 
                onClick={() => handleAnswer('real')} 
                className="py-8 bg-emerald-500 hover:bg-emerald-600 text-white rounded-3xl text-3xl font-black transition-all active:scale-95 shadow-lg shadow-emerald-100"
              >
                REAL
              </button>
              <button 
                onClick={() => handleAnswer('fake')} 
                className="py-8 bg-rose-500 hover:bg-rose-600 text-white rounded-3xl text-3xl font-black transition-all active:scale-95 shadow-lg shadow-rose-100"
              >
                FAKE
              </button>
            </div>
          </div>
        ) : (
          /* 결과 화면 */
          <div className="space-y-8 animate-in fade-in duration-700">
            <div className="bg-white rounded-[40px] shadow-sm border border-slate-100 p-12 text-center">
              <h2 className="text-4xl font-black text-slate-900 mb-10">최종 대결 결과</h2>
              <div className="grid grid-cols-2 gap-8 mb-10">
                <div className="p-8 bg-blue-50/50 rounded-[32px] border border-blue-100">
                  <p className="text-blue-500 font-bold mb-2 uppercase text-xs tracking-widest">User Score</p>
                  <p className="text-6xl font-black text-blue-900">{results.userCorrect}</p>
                </div>
                <div className="p-8 bg-indigo-50/50 rounded-[32px] border border-indigo-100">
                  <p className="text-indigo-500 font-bold mb-2 uppercase text-xs tracking-widest">AI Score</p>
                  <p className="text-6xl font-black text-indigo-900">{results.aiCorrect}</p>
                </div>
              </div>
              <div className="py-5 px-12 bg-slate-900 text-white rounded-2xl inline-block text-2xl font-bold shadow-xl">
                {results.userCorrect > results.aiCorrect ? "🏆 인간이 AI를 이겼습니다!" : 
                 results.userCorrect < results.aiCorrect ? "🤖 AI의 분석이 더 정확했습니다!" : "🤝 우열을 가릴 수 없는 무승부!"}
              </div>
            </div>

            {/* 상세 리포트 테이블 - 글씨 및 레이아웃 확장 */}
            <div className="bg-white rounded-[40px] shadow-sm border border-slate-100 overflow-hidden">
              <div className="p-10 bg-slate-50 border-b border-slate-100">
                <h3 className="font-black text-slate-800 text-2xl text-center">상세 대조 리포트</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="text-slate-400 text-sm font-black uppercase tracking-widest border-b border-slate-100">
                      <th className="px-10 py-8 text-center">이미지</th>
                      <th className="px-10 py-8">나의 예측</th>
                      <th className="px-10 py-8">AI 분석 결과</th>
                      <th className="px-10 py-8 text-center">정답</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {results.details.map((item, idx) => (
                      <tr key={idx} className="hover:bg-slate-50/50 transition-colors">
                        <td className="px-10 py-8">
                          <img 
                            src={item.img} 
                            className="w-28 h-28 rounded-3xl object-cover border-2 border-slate-100 mx-auto shadow-sm" 
                            alt="res" 
                          />
                        </td>
                        <td className="px-10 py-8">
                          <p className={`text-2xl font-black ${item.userIsRight ? 'text-emerald-500' : 'text-rose-500'}`}>
                            {item.userPick.toUpperCase()}
                            <span className="ml-2 text-xl">{item.userIsRight ? '✓' : '✗'}</span>
                          </p>
                        </td>
                        <td className="px-10 py-8">
                          <p className={`font-black text-xl mb-1 ${item.aiIsRight ? 'text-indigo-600' : 'text-rose-400'}`}>
                            {item.aiVerdict.toUpperCase()}
                          </p>
                          <div className="flex items-center gap-2">
                            <div className="w-24 h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div 
                                    className={`h-full ${item.aiIsRight ? 'bg-indigo-400' : 'bg-rose-300'}`} 
                                    style={{ width: `${item.aiScore}%` }}
                                ></div>
                            </div>
                            <span className="text-sm text-slate-400 font-bold">{item.aiScore}%</span>
                          </div>
                        </td>
                        <td className="px-10 py-8 text-center">
                          <span className="inline-block bg-slate-900 text-white px-6 py-3 rounded-2xl text-base font-black tracking-tighter">
                            {item.label.toUpperCase()}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            
            <button 
              onClick={() => navigate('/')} 
              className="w-full py-8 bg-white border-2 border-slate-200 text-slate-500 rounded-[32px] text-2xl font-black hover:bg-slate-50 hover:border-slate-300 transition-all shadow-sm"
            >
              홈페이지로 나가기
            </button>
          </div>
        )}
      </div>
    </div>
  );
}