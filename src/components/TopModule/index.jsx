/*
 * @Date: 2022-05-14 20:45:48
 * @LastEditors: JZY
 * @LastEditTime: 2023-02-05 16:26:33
 * @FilePath: /project/Visual/src/components/TopModule/index.jsx
 */
import React, { Component } from "react";
import axios from "axios";
import {Link} from 'react-router-dom'
import { Typography, Row, Col, Select, Button, Tooltip, Tag ,Space,Image,Modal} from "antd";
import logo from './logo.png'
import {
  GithubOutlined,
  QuestionCircleOutlined,
  ShareAltOutlined,
  SyncOutlined
} from "@ant-design/icons";

import "./index.css";

const { Option } = Select;
const { Text, Title } = Typography;

const info = [
  ["Dataset:", ["breastCancer"], 0],
  ["WSI Count:", ["3"], 0],
  ["Backbone:", ["ResNet50"], 0],
  ["Scatter:", ["TSNE", "UMAP"], 0],

];

export default class TopModule extends Component {
  constructor(props) {
    super(props);
    this.state = {
      img_len: Object.keys(this.props.WSI_Data.img_id).length,
      visible:false
    };
  }
  componentDidMount() {
    const savedPatientName = localStorage.getItem('patientName');
    const savedPatientID = localStorage.getItem('patientID');
    
      this.setState({
        savedPatientName:savedPatientName,
        savedPatientID :savedPatientID
      });
    }
  toGithub = () => {
    window.open("https://github.com/mad01der")
  }
  WsiUpload = () => {
    // 打印log日志
    console.log("Upload WSI")
  }
  showModal = () => {
    this.setState({
      visible: true,
    });
  };

  handleOk = () => {
    this.setState({
      visible: false,
    });
  };

  handleCancel = () => {
    this.setState({
      visible: false,
    });
  };


  render() {
    return (
      <>
        <Row>
          <Col span={1}>
            <Image style="small"
              fallback="data:image/png;base64,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"
              src={logo}
              style={{ maxWidth: '90%', height: 'auto' }}
            />
          </Col>
          <Col span={7}>
            <Title>PathoVision</Title>
            
          </Col>
          <Col span={1}>
            <Button className="e-button" type="primary" onClick = {this.showModal}>说明</Button>
          </Col>

          {/* 医生信息 */}
          <Row offset={1} span={7}>
            <Space direction="horizontal" style={{ marginTop: '0px' }}>
              <Text className="siderFont">医生姓名：</Text>
              <Text className="siderFont">Tom</Text>
              <Text className="siderFont">医生号码：</Text>
              <Text className="siderFont">1934567</Text>
            </Space>
            <Space  direction="horizontal">
              <Text className="siderFont">就诊号:</Text>
              <Text className="siderFont">{this.state.savedPatientName}</Text>
              <Text className="siderFont">报告号:</Text>
              <Text className="siderFont">{this.state.savedPatientID}</Text>
            </Space>
          </Row>

          {/* Github 图标 */}
          <Col span={6} style={{ textAlign: 'right' }}>
            <Tooltip placement="bottomRight" title="Click the icon to jump to Github page.">
              <Button type="text" className="siderFont" shape="circle" icon={<GithubOutlined />} onClick={this.toGithub} />
            </Tooltip>
            <Link to="../Data/">
              <Button className="e-button" type="primary">返回病人列表页面</Button>
            </Link>
          </Col>
        </Row>
        <Modal
          title="说明"
          visible={this.state.visible}
          onOk={this.handleOk}
          onCancel={this.handleCancel}
        >
          <p>由于病理图像切片数量多，标注成本高,所以我们将切片进行分类和预测诊断</p>
          <p>在此过程中,由于可能会存在切片图像不清晰,或者勿标注等情况</p>
          <p>所以需要进行噪声标签的检测,我们将噪声标签列出并给医生提供人工标注途径</p>
          <p>下面是对BMM和PR检测指标的一些说明。</p>
          <p>BMM指标是使用Beta混合分布拟合噪声和干净样本的方法</p>
          <p>BMM的值越小,说明样本是噪声样本的可能性越低</p>
          <p>反之,若BMM的值越大,说明样本是噪声样本的可能性越高</p>
          <p>PR是使用惩罚线性回归方法进行检测</p>
          <p>PR的值表示样本偏移的值</p>
          <p>值越大，说明样本是噪声的可能性越大</p>

        </Modal>
      </>
    
    );
  }
}
