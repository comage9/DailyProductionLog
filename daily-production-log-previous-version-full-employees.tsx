import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Clock, BarChart2, Cog, Shapes, Hash } from 'lucide-react';

const MultilingualHeader = ({ icon, korean, thai, vietnamese }) => (
  <div className="flex items-center">
    {icon}
    <div className="ml-2">
      <h2 className="text-xl font-semibold">{korean}</h2>
      <p className="text-sm text-gray-500">{thai} / {vietnamese}</p>
    </div>
  </div>
);

const employees = [
  { id: 1, name: '김주현', workHours: '주간', machineNumber: '' },
  { id: 2, name: '이정우', workHours: '주간', machineNumber: '' },
  { id: 3, name: '이경란', workHours: '주간', machineNumber: '' },
  { id: 4, name: '(공석)', workHours: '', machineNumber: '' },
  { id: 5, name: '(공석)', workHours: '', machineNumber: '' },
  { id: 6, name: '요이', workHours: '주간', machineNumber: '' },
  { id: 7, name: '덕', workHours: '주간', machineNumber: '' },
  { id: 8, name: '황', workHours: '주간', machineNumber: '' },
  { id: 9, name: '키엔', workHours: '주간', machineNumber: '' },
  { id: 10, name: '(공석)', workHours: '', machineNumber: '' },
  { id: 11, name: '(공석)', workHours: '', machineNumber: '' },
  { id: 12, name: '노경열', workHours: '주간', machineNumber: '15' },
  { id: 13, name: '다이', workHours: '주간', machineNumber: '' },
  { id: 14, name: '홍', workHours: '주간', machineNumber: '' },
  { id: 15, name: '특', workHours: '주간', machineNumber: '' },
  { id: 16, name: '후이', workHours: '주간', machineNumber: '7' },
  { id: 17, name: '떼', workHours: '야간', machineNumber: '' },
  { id: 18, name: '깨', workHours: '야간', machineNumber: '' },
  { id: 19, name: '톰', workHours: '야간', machineNumber: '' },
  { id: 20, name: '나우', workHours: '야간', machineNumber: '' },
  { id: 21, name: '쏭', workHours: '야간', machineNumber: '' },
  { id: 22, name: '또', workHours: '야간', machineNumber: '4' },
  { id: 23, name: '팽', workHours: '야간', machineNumber: '8' },
  { id: 24, name: '웃', workHours: '야간', machineNumber: '5-6' },
  { id: 25, name: '앙', workHours: '주간', machineNumber: '13-14' },
  { id: 26, name: '안태욱', workHours: '주간', machineNumber: '11-12' },
  { id: 27, name: '(공석)', workHours: '', machineNumber: '' },
  { id: 28, name: '꽝안', workHours: '주간', machineNumber: '9-10' },
  { id: 29, name: '(공석)', workHours: '', machineNumber: '' },
  { id: 30, name: '(공석)', workHours: '', machineNumber: '' },
  { id: 31, name: '샌디', workHours: '주간', machineNumber: '' },
  { id: 32, name: '뿌이', workHours: '야간', machineNumber: '7' },
  { id: 33, name: '묘우', workHours: '주간', machineNumber: '' },
];

const DailyProductionLog = () => {
  const [formData, setFormData] = useState({
    year: new Date().getFullYear().toString(),
    startTime: '',
    endTime: '',
    employeeNumber: '',
    machineNumber: '',
    moldNumber: '',
    productionQuantity: '',
    specialNote: '',
  });

  const [isRunning, setIsRunning] = useState(false);
  const [startDate, setStartDate] = useState(null);

  useEffect(() => {
    let interval;
    if (isRunning) {
      interval = setInterval(() => {
        const now = new Date();
        setFormData(prevState => ({
          ...prevState,
          startTime: startDate.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }),
          endTime: now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
        }));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRunning, startDate]);

  const handleStartStop = () => {
    if (!isRunning) {
      setStartDate(new Date());
      setIsRunning(true);
    } else {
      setIsRunning(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));

    if (name === 'employeeNumber') {
      const employee = employees.find(emp => emp.id.toString() === value);
      if (employee) {
        setFormData(prevState => ({
          ...prevState,
          machineNumber: employee.machineNumber
        }));
      }
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Form submitted:', formData);
    // Here you would typically send the data to your backend
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">
        일일 생산 일지
        <br />
        <span className="text-lg font-normal">บันทึกการผลิตประจำวัน / Nhật ký sản xuất hàng ngày</span>
      </h1>
      <form onSubmit={handleSubmit}>
        <Card className="mb-4">
          <CardHeader>
            <MultilingualHeader
              icon={<Clock className="mr-2" />}
              korean="년도 및 시간"
              thai="ปีและเวลา"
              vietnamese="Năm và thời gian"
            />
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <Input
                type="number"
                name="year"
                value={formData.year}
                onChange={handleInputChange}
                placeholder="년도"
                className="w-full"
              />
              <Button type="button" onClick={handleStartStop} className="w-full">
                {isRunning ? '정지' : '시작'}
              </Button>
              <Input
                type="text"
                name="startTime"
                value={formData.startTime}
                readOnly
                placeholder="시작 시간"
                className="w-full"
              />
              <Input
                type="text"
                name="endTime"
                value={formData.endTime}
                readOnly
                placeholder="종료 시간"
                className="w-full"
              />
            </div>
          </CardContent>
        </Card>

        <Card className="mb-4">
          <CardHeader>
            <MultilingualHeader
              icon={<Hash className="mr-2" />}
              korean="사원 번호"
              thai="รหัสพนักงาน"
              vietnamese="Mã nhân viên"
            />
          </CardHeader>
          <CardContent>
            <Select
              name="employeeNumber"
              value={formData.employeeNumber}
              onValueChange={(value) => handleInputChange({ target: { name: 'employeeNumber', value } })}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="사원 번호를 선택하세요" />
              </SelectTrigger>
              <SelectContent>
                {employees.map((employee) => (
                  <SelectItem key={employee.id} value={employee.id.toString()}>
                    {employee.id} - {employee.name} ({employee.workHours})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardContent>
        </Card>

        <Card className="mb-4">
          <CardHeader>
            <MultilingualHeader
              icon={<Cog className="mr-2" />}
              korean="기계 번호"
              thai="หมายเลขเครื่องจักร"
              vietnamese="Số máy"
            />
          </CardHeader>
          <CardContent>
            <Select
              name="machineNumber"
              value={formData.machineNumber}
              onValueChange={(value) => handleInputChange({ target: { name: 'machineNumber', value } })}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="기계 번호를 선택하세요" />
              </SelectTrigger>
              <SelectContent>
                {[...Array(16)].map((_, i) => (
                  <SelectItem key={i + 1} value={(i + 1).toString()}>
                    {i + 1}번 기계
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input
              name="machineNumber"
              value={formData.machineNumber}
              onChange={handleInputChange}
              placeholder="또는 기계 번호를 직접 입력하세요"
              className="w-full mt-2"
            />
          </CardContent>
        </Card>

        <Card className="mb-4">
          <CardHeader>
            <MultilingualHeader
              icon={<Shapes className="mr-2" />}
              korean="금형 번호"
              thai="หมายเลขแม่พิมพ์"
              vietnamese="Số khuôn"
            />
          </CardHeader>
          <CardContent>
            <Input
              name="moldNumber"
              value={formData.moldNumber}
              onChange={handleInputChange}
              placeholder="금형 번호를 입력하세요"
              className="w-full"
            />
          </CardContent>
        </Card>

        <Card className="mb-4">
          <CardHeader>
            <MultilingualHeader
              icon={<BarChart2 className="mr-2" />}
              korean="생산 수량"
              thai="ปริมาณการผลิต"
              vietnamese="Số lượng sản xuất"
            />
          </CardHeader>
          <CardContent>
            <Input
              name="productionQuantity"
              value={formData.productionQuantity}
              onChange={handleInputChange}
              placeholder="생산 수량을 입력하세요"
              className="w-full"
              type="number"
            />
          </CardContent>
        </Card>

        <Card className="mb-4">
          <CardHeader>
            <MultilingualHeader
              icon={<img src="/api/placeholder/50/50" alt="Today only" className="mr-2" />}
              korean="오늘의 특이사항"
              thai="บันทึกพิเศษของวันนี้"
              vietnamese="Ghi chú đặc biệt hôm nay"
            />
          </CardHeader>
          <CardContent>
            <Textarea
              name="specialNote"
              value={formData.specialNote}
              onChange={handleInputChange}
              placeholder="오늘의 특이사항을 입력하세요"
              className="w-full"
            />
          </CardContent>
        </Card>

        <Button type="submit" className="w-full">
          제출 / ส่ง / Gửi
        </Button>
      </form>
    </div>
  );
};

export default DailyProductionLog;
