import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

// 이미지 임포트 (데이터가 많아지면 public 폴더를 사용하는 것이 효율적이지만, 현재 구조를 유지합니다)
// 예시로 50개 리스트를 작성합니다. 실제 파일이 있는 만큼 추가해주세요.
// Fake Images Import
import fake01 from '../game_image/fake_01.png';
import fake02 from '../game_image/fake_02.png';
import fake03 from '../game_image/fake_03.png';
import fake04 from '../game_image/fake_04.png';
import fake05 from '../game_image/fake_05.png';
import fake06 from '../game_image/fake_06.png';
import fake07 from '../game_image/fake_07.png';
import fake08 from '../game_image/fake_08.png';
import fake09 from '../game_image/fake_09.png';
import fake10 from '../game_image/fake_10.png';
import fake11 from '../game_image/fake_11.png';
import fake12 from '../game_image/fake_12.png';
import fake13 from '../game_image/fake_13.png';
import fake14 from '../game_image/fake_14.png';
import fake15 from '../game_image/fake_15.png';
import fake16 from '../game_image/fake_16.png';
import fake17 from '../game_image/fake_17.png';
import fake18 from '../game_image/fake_18.png';
import fake19 from '../game_image/fake_19.png';
import fake20 from '../game_image/fake_20.png';

// Real Images Import
import real01 from '../game_image/real_01.png';
import real02 from '../game_image/real_02.png';
import real03 from '../game_image/real_03.png';
import real04 from '../game_image/real_04.png';
import real05 from '../game_image/real_05.png';
import real06 from '../game_image/real_06.png';
import real07 from '../game_image/real_07.png';
import real08 from '../game_image/real_08.png';
import real09 from '../game_image/real_09.png';
import real10 from '../game_image/real_10.png';
import real11 from '../game_image/real_11.png';
import real12 from '../game_image/real_12.png';
import real13 from '../game_image/real_13.png';
import real14 from '../game_image/real_14.png';
import real15 from '../game_image/real_15.png';
import real16 from '../game_image/real_16.png';
import real17 from '../game_image/real_17.png';
import real18 from '../game_image/real_18.png';
import real19 from '../game_image/real_19.png';
import real20 from '../game_image/real_20.png';

// 전체 데이터 풀 (25개씩 총 50개라고 가정)
const ALL_FAKE_DATA = [
  { img: fake01, label: 'fake', aiScore: 18 },
  { img: fake02, label: 'fake', aiScore: 37 },
  { img: fake03, label: 'fake', aiScore: 66 },
  { img: fake04, label: 'fake', aiScore: 47 },
  { img: fake05, label: 'fake', aiScore: 41 },
  { img: fake06, label: 'fake', aiScore: 35 },
  { img: fake07, label: 'fake', aiScore: 27 },
  { img: fake08, label: 'fake', aiScore: 90 },
  { img: fake09, label: 'fake', aiScore: 71 },
  { img: fake10, label: 'fake', aiScore: 21 },
  { img: fake11, label: 'fake', aiScore: 8 },
  { img: fake12, label: 'fake', aiScore: 39 },
  { img: fake13, label: 'fake', aiScore: 12 },
  { img: fake14, label: 'fake', aiScore: 15 },
  { img: fake15, label: 'fake', aiScore: 18 },
  { img: fake16, label: 'fake', aiScore: 13 },
  { img: fake17, label: 'fake', aiScore: 20 },
  { img: fake18, label: 'fake', aiScore: 48 },
  { img: fake19, label: 'fake', aiScore: 19 },
  { img: fake20, label: 'fake', aiScore: 60 },
  // ... 나머지 20개도 객체 형태로 추가
];

const ALL_REAL_DATA = [
  { img: real01, label: 'real', aiScore: 85 },
  { img: real02, label: 'real', aiScore: 77 },
  { img: real03, label: 'real', aiScore: 18 },
  { img: real04, label: 'real', aiScore: 85 },
  { img: real05, label: 'real', aiScore: 81 },
  { img: real06, label: 'real', aiScore: 69 },
  { img: real07, label: 'real', aiScore: 88 },
  { img: real08, label: 'real', aiScore: 99 },
  { img: real09, label: 'real', aiScore: 7 },
  { img: real10, label: 'real', aiScore: 89 },
  { img: real11, label: 'real', aiScore: 84 },
  { img: real12, label: 'real', aiScore: 35 },
  { img: real13, label: 'real', aiScore: 54 },
  { img: real14, label: 'real', aiScore: 90 },
  { img: real15, label: 'real', aiScore: 68 },
  { img: real16, label: 'real', aiScore: 28 },
  { img: real17, label: 'real', aiScore: 79 },
  { img: real18, label: 'real', aiScore: 92 },
  { img: real19, label: 'real', aiScore: 88 },
  { img: real20, label: 'real', aiScore: 90 },
  // ... 나머지 20개도 객체 형태로 추가
];

export default function GamePage() {
  const navigate = useNavigate();
  
  const [gameData, setGameData] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [userAnswers, setUserAnswers] = useState([]);
  const [showResult, setShowResult] = useState(false);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    // 1. Fake에서 5개 랜덤 추출
    const shuffledFake = [...ALL_FAKE_DATA].sort(() => Math.random() - 0.5).slice(0, 5);
    // 2. Real에서 5개 랜덤 추출
    const shuffledReal = [...ALL_REAL_DATA].sort(() => Math.random() - 0.5).slice(0, 5);
    
    // 3. 합쳐서 다시 전체 랜덤 셔플
    const finalRound = [...shuffledFake, ...shuffledReal].sort(() => Math.random() - 0.5);
    
    setGameData(finalRound);
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
              >REAL</button>
              <button 
                onClick={() => handleAnswer('fake')} 
                className="py-8 bg-rose-500 hover:bg-rose-600 text-white rounded-3xl text-3xl font-black transition-all active:scale-95 shadow-lg shadow-rose-100"
              >FAKE</button>
            </div>
          </div>
        ) : (
          <div className="space-y-8 animate-in fade-in duration-700">
            {/* 결과 요약 카드 */}
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

            {/* 상세 리포트 테이블 */}
            <div className="bg-white rounded-[40px] shadow-sm border border-slate-100 overflow-hidden">
              <div className="p-10 bg-slate-50 border-b border-slate-100 text-center">
                <h3 className="font-black text-slate-800 text-2xl">상세 대조 리포트</h3>
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
                          <img src={item.img} className="w-28 h-28 rounded-3xl object-cover border-2 border-slate-100 mx-auto shadow-sm" alt="res" />
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
                                <div className={`h-full ${item.aiIsRight ? 'bg-indigo-400' : 'bg-rose-300'}`} style={{ width: `${item.aiScore}%` }}></div>
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
              className="w-full py-8 bg-white border-2 border-slate-200 text-slate-500 rounded-[32px] text-2xl font-black hover:bg-slate-50 transition-all shadow-sm"
            >홈페이지로 나가기</button>
          </div>
        )}
      </div>
    </div>
  );
}