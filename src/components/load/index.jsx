import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import { Card, Button, Input, Upload, Progress, Col,Typography} from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import "./load.css";
import axios from "axios";
import { ArrowRightOutlined } from '@ant-design/icons';
import logo from './logo.jpg';
import background from './background.jpg';
const { Text, Title } = Typography;

export default class Load extends Component {
    constructor(props) {
        super(props);
        this.state = {
            patientName: '',
            patientID: '',
            fileList: [],
            progress: 0,
            trainingComplete: false,
            showResultButton: false,   // 是否显示查看结果按钮
            print_load: ''
        };
        this.timer = null;
    }
    componentDidMount() {

        this.timer = setInterval(this.fetchProgress, 1000); // 每秒轮询一次后端
    }
    componentWillUnmount() {
        clearInterval(this.timer); // 组件卸载时清除定时器
    }
    fetchProgress = () => {
        axios.get('http://127.0.0.1:5000/loading_2')
            .then(response => {
                console.log(response)
                const progress = response.data.progress;
                const print_load = response.data.print_load;
                const trainingComplete = response.data.trainingComplete;
                this.setState({
                    print_load:print_load
                })
                this.setState({ progress, trainingComplete});
                if (trainingComplete) {
                    clearInterval(this.timer); // 训练完成后停止轮询
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });

    };
    click = () => {
        this.props.history.replace('/Data');
    };
    handleInputChange = (e) => {
        this.setState({ [e.target.name]: e.target.value });
    };

    handleFileChange = (info) => {
        this.setState({ fileList: [...info.fileList] });
    };

    handleSubmit = (e) => {

        const { patientName, patientID, fileList } = this.state;

        // 构造 FormData 对象
        const formData = new FormData();
        formData.append('patientName', patientName);
        formData.append('patientID', patientID);
        formData.append('file', fileList[0].originFileObj); // 获取文件对象
        localStorage.setItem('patientName', patientName);
        localStorage.setItem('patientID', patientID);
        this.setState({ trainingInProgress: true, showResultButton: false });
        
        this.fetchProgress();

        axios.post('http://127.0.0.1:5000/load', formData, {
            headers: {
                'Content-Type': 'multipart/form-data' // 设置请求头为 multipart/form-data
            }
        })
            .then(response => {
                console.log('Response from server:', response.data);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        axios.post('http://127.0.0.1:5000/mysql', formData, {
            headers: {
                'Content-Type': 'multipart/form-data' // 设置请求头为 multipart/form-data
            }
        })
            .then(response => {
                console.log('Response from server:', response.data);
            })
            .catch(error => {
                console.error('Error:', error);
            });
    };
    back = () =>{
      this.props.history.replace('/Data');
    }
    handleReset = () => {
       window.location.reload()
    };
    render() {
        const { progress, trainingComplete, trainingInProgress } = this.state;
        return (
            <div className="page-container" style={{ backgroundImage: `url(${background})`,opacity: 0.9}}>
              <img className = "card_pic" src={logo} alt="Logo" style={{ width: '500px', height: '150px', position: 'absolute', top: 50, left: 760, marginLeft: 0, marginTop: 0 }} />
            <div className="return-button-container" style={{ position: 'absolute', top: '50px', left: '50px' }}>
              <Button type="primary" onClick={this.back} >返回</Button>
            </div>
            {/* <div className="return-button-container" style={{ position: 'absolute', top: '50px', right: '50px' }}>
              <Button type="primary" onClick={this.click} >查看已有结果</Button>
            </div> */}
              <Card className="card-wrapper">
                <Col >
                  <form onSubmit={this.handleSubmit}>
                    <div className="card-content">
                      <label>就诊号：</label>
                      <Input name="patientName" onChange={this.handleInputChange} />
                    </div>
                    <div className="card-content">
                      <label>报告号：</label>
                      <Input name="patientID" onChange={this.handleInputChange} />
                    </div>
                    <div className="card-content">
                      <label>选择文件：</label>
                      <Upload onChange={this.handleFileChange} fileList={this.state.fileList}>
                        <Button icon={<UploadOutlined />}>选择文件</Button>
                      </Upload>
                    </div>
                    <div className="card-buttons">
                      <Button type="primary" onClick={this.handleSubmit}>确认</Button>
                      <Button onClick={this.handleReset}>取消</Button>
                    </div>
                  </form>
                </Col>
              </Card>
              <Card className="progress-container">
                <Text style={{ fontWeight: 'bold', color: 'black' }}>训练细节</Text>
                <Col>
                
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
                  <div style={{ marginBottom: '5px' }}>
                    <span style={{ fontWeight: 'bold' }}>进度显示框：</span>
                  </div>
                  <div>
                    <textarea
                      value={this.state.print_load}
                      readOnly
                      style={{
                        width: '380px',
                        height: '50px',
                        padding: '5px',
                        border: '1px solid #ccc',
                        borderRadius: '5px',
                        resize: 'none' // 禁止用户调整文本框大小
                      }}
                    />
                  </div>
                </div>
                <div className="progress-bar">
                <Text style={{ fontWeight: 'bold', marginBottom: '5px' }}>进度条：</Text>
                  <Progress percent={progress} size={[300, 50]} status={trainingComplete ? "success" : "active" } />
                </div>
                  {trainingComplete &&  <Button className="button" type="primary" onClick={this.click}> 查看结果</Button>}
                </Col>
              </Card>
            </div>
          );  
    }
}


