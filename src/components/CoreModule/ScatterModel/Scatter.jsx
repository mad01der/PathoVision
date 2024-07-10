import * as d3 from "d3";
import React, { Component } from "react";
import ReactECharts from "echarts-for-react";
import { PlusOutlined } from '@ant-design/icons';

import {
  Row,
  Col,
  Typography,
  Select,
  Radio,
  Slider,
  Menu,
  Dropdown,
  Button,
  Descriptions,
  Card,
  Image,
  Switch,
  Modal,
  Affix,Tag
} from "antd";

import {
  FilterOutlined,
  UnorderedListOutlined,
  ColumnWidthOutlined,
  TagOutlined,
} from "@ant-design/icons";
import "./index.css";
import axios from "axios";
const { Option } = Select;
const { Text } = Typography;
const labelStyle = { fontWeight: 500 };
const category_name = ["no cancer", "cancer", "high cancer"];
const category_color = ["#40b373", "#d4b446", "#ee8826"]; // and more
export default class Scatter extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isModalVisible:false,
      selectedWSI: 0,
      result:null,
      category: -1, // 筛选类别
      option: {},
      noiseChild: [], 
      dataToShow:[],
      datacolor:null,
      selectedOption: null, // 保存选择的选项
      selectedImage: null ,
      path :null,
      path2:null,
      file_name:[],
      load: true,
      visible: false,
      confirmLoading: false,
      barOption: {},
      areaOption: {},
      drawerVisible: false,
      recentclass:null,
      totalnum:0
    };
  }
  componentDidMount() {
    this.newFunction();
    this.newFunction2("无癌症"); 
  };
  setclass = async(args,recent) =>{
    await this.setState({
      recentclass:recent
    });
  }
  setdraw = async (args, bmm,spr,bmms,sprs) => {
    await this.setState({
      barOption: {
        grid: {
          left: "30%",
          right: "10%",
        },
        yAxis: {
          type: "category",
          data: ["Bmm", "PR"],
        },
        xAxis: {
          type: "value",
          min: 0.0,
          max: 1.0,
          show: false,
          splitLine: {
            //网格线
            show: false,
          },
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "shadow",
          },
        },
        series: [
          {
            data: [{
              value: bmm,
              itemStyle: {
                color: "#5470c6"
              }
            },
            {
              value: spr,
              itemStyle: {
                color: "#93cd77"
              }
            }
            ],
            type: "bar",
          },
        ],
      },
      areaOption: {
        grid: {
          top: 0,
          left: "13%",
          right: "13%",
          bottom: 20,
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "cross",
            animation: false,
            label: {
              backgroundColor: "#505765",
            },
          },
        },
        xAxis: [
          {
            type: "category",
            boundaryGap: false,
            axisLine: { onZero: false },
            // prettier-ignore
            data: 1,
          },
        ],
        yAxis: [
          {
            name: "Bmm",
            type: "value",
            min: 0,
            max: 1,
          },
          {
            name: "PR",
            nameLocation: "start",
            alignTicks: true,
            type: "value",
            inverse: true,
            min: 0,
            max: 1,
          },
        ],
        series: [
          {
            name: "Bmm",
            type: "line",
            areaStyle: {},
            lineStyle: {
              width: 1,
            },
            showSymbol: false,
            emphasis: {
              focus: "series",
            },
            markArea: {
              silent: true,
              itemStyle: {
                opacity: 0.3,
              },
            },
            data: bmms,
          },
          {
            name: "Spr",
            type: "line",
            yAxisIndex: 1,
            showSymbol: false,
            areaStyle: {},
            lineStyle: {
              width: 1,
            },
            emphasis: {
              focus: "series",
            },
            markArea: {
              silent: true,
              itemStyle: {
                opacity: 0.3,
              },
            },
            data: sprs,
          },
        ],
      },
    });
    this.setState({
      visible: args,
    });
  };
  handleWSIClick = (index) => {
    // 处理点击事件并将值传出
    this.setState({ 
      selectedWSI: index 
    });
    console.log(this.state.selectedWSI)
  };
  handleCancel = () => {
    this.setState({
      visible: false,
    });
  };
  newFunction = () => {
    var This = this;
    var sample_data= [];
    var noiseChild = [];
    var data_len = Object.keys(this.props.sample_Data.class).length;
    console.log(data_len)
    for (var i = 0; i < data_len; i++) {
        sample_data.push([
          This.props.sample_Data.scatter_x[i],
          This.props.sample_Data.scatter_y[i],
          This.props.sample_Data.patch_id[i],
          This.props.sample_Data.img_id[i],
          This.props.sample_Data.class[i],
          // new
          This.props.sample_Data.bmm_num[i],
          This.props.sample_Data.spr_num[i],
          This.props.sample_Data.file_name[i],
          This.props.sample_Data.kmeans_label[i],
          This.props.sample_Data.file_name[i],
        ]);
        if (This.props.sample_Data.noise[i] > 0 && This.props.sample_Data.img_id[i] == this.state.selectedWSI) {
          noiseChild.push(This.props.sample_Data.file_name[i]);
      }
      this.setState({ 
        noiseChild: noiseChild
       }); // 更新状态以显示噪声子项
    }
  };
  showDetails = () => {
    this.setState({ isModalVisible: true });
  };
  newFunction2 = (type) => {
    var This = this;
    var sample_data= [];
    var greenData=[];
    var yellowData=[];
    var redData =[];
    var dataToShow = [];
    var datacolor = null;
    var data_len = Object.keys(this.props.sample_Data.class).length;
    console.log("here ")
    console.log(data_len)
    for (var i = 0; i < data_len; i++) {
        sample_data.push([
          This.props.sample_Data.scatter_x[i],
          This.props.sample_Data.scatter_y[i],
          This.props.sample_Data.patch_id[i],
          This.props.sample_Data.img_id[i],
          This.props.sample_Data.class[i],
          // new
          This.props.sample_Data.bmm_num[i],
          This.props.sample_Data.spr_num[i],
          This.props.sample_Data.file_name[i],
          This.props.sample_Data.kmeans_label[i],
          This.props.sample_Data.file_name[i],
        ]);
        if (This.props.sample_Data.class[i] == 1 && This.props.sample_Data.img_id[i] == this.state.selectedWSI) {
          greenData.push(This.props.sample_Data.file_name[i]);
        }
        if (This.props.sample_Data.class[i] == 2 && This.props.sample_Data.img_id[i] == this.state.selectedWSI) {
          yellowData.push(This.props.sample_Data.file_name[i]);
        }
        if (This.props.sample_Data.class[i] == 3 && This.props.sample_Data.img_id[i] == this.state.selectedWSI) {
          redData.push(This.props.sample_Data.file_name[i]);
        }
    }
     switch (type) {
      case '无癌症':
        dataToShow = greenData;
        datacolor = '#40b373'
        break;
      case '有癌症':
        dataToShow = yellowData;
        datacolor = '#d4b446'
        break;
      case '高癌症':
        dataToShow = redData;
        datacolor = '#ee8826'
        break;
      default:
        break;
    }
    this.setState({ dataToShow:dataToShow,datacolor:datacolor});
  };
  openDraw = () =>{
    this.setState({ visible: true });
  };
  setConfirmLoading = (args) => {
    this.setState({
      confirmLoading: args,
    });
  };
  handleOk = () => {
    var T = this;
    this.setConfirmLoading(true);
    setTimeout(() => {
      this.setState({visible:false});
      this.setConfirmLoading(false);
      window.location.reload()
    }, 2000);
  };
  handleSelectChange = (value) => {
    this.setState({ selectedOption: value }, () => {
      // 在选项变化后根据选项值更新图片地址
      this.updateSelectedImage();
    });
    // console.log("Here is the result " + value + this.state.selectedWSI)
    // this.setVisible(true, value);
    var bmm = 0;
    var spr = 0;
    var bmms = 0;
    var sprs = 0;
    var recent = null;
    var data_len = Object.keys(this.props.sample_Data.class).length;
    for(var i = 0;i< data_len;i++)
    {
      if(this.props.sample_Data.img_id[i] == this.state.selectedWSI && this.props.sample_Data.file_name[i] == value){
        bmm = this.props.sample_Data.bmm[i];
        bmms = this.props.sample_Data.bmm_num[i];
        spr = this.props.sample_Data.spr[i];
        sprs = this.props.sample_Data.spr_num[i];
        recent = this.props.sample_Data.class[i];
        this.setState({
            path:this.props.sample_Data.patch_id[i],
            path2:this.props.sample_Data.img_id[i]
        });
      }
    }
    this.setclass(true,recent);
    // console.log("recent class " + this.state.recentclass);
    this.setdraw(true,bmm,spr,bmms,sprs);
    };
  updateSelectedImage = () => {
    console.log(this.state.selectedOption)
    var temp = this.state.selectedOption;
    var result = temp.split("_").slice(0, 4).join("_") + ".png";
    this.setState({
      result:result
    });
  };
  handleSelectChange_2 = (value) => {
    const selectedLabel = category_name[value];
    console.log("Selected Label: ++++++++++++++", selectedLabel);
    console.log("patch_id" ,this.state.path)
    console.log("img_id" ,this.state.path2)
    axios
     .post('http://127.0.0.1:5000/last', {
      selectedLabel: selectedLabel,
      path: this.state.path,
      path2: this.state.path2
      })
      .then(response => {
        console.log('Response from server:', response.data);
      })
      .catch(error => {
        console.error('Error:', error);
      });
      
}
  getcolor = () =>{
    console.log( this.state.datacolor);
  }
  render() {
    return (
      <>
        <Modal
          title="细节信息"
          visible={this.state.isModalVisible}
          onCancel={() => this.setState({ isModalVisible: false })}
          footer={[
            <Button key="close" onClick={() => this.setState({ isModalVisible: false })}>关闭</Button>
          ]}
          style={{ top: '50%', transform: 'translateY(-50%)' }} // 将Modal居中显示
        >
          <div style={{ display: 'flex', flexWrap: 'wrap' }}>
            {this.state.dataToShow.map((imageUrl, index) => (
              <div key={index} style={{ width: '12.5%', marginBottom: '20px', padding: '0 10px' }}>
                <img src={process.env.REACT_APP_IMAGE_PATH_NEW + "/image_" + this.state.selectedWSI + "/" + imageUrl.split("_").slice(0, 4).join("_") + ".png"}  style={{ maxWidth: '100%', height: 'auto' }} />
                <div>{imageUrl.split("_").slice(0, 4).join("_") + ".png"}</div>
              </div>
            ))}
            {this.state.dataToShow.length % 8 !== 0 && (
              // 添加占位元素以保持每行显示 7 张图片
              <div style={{ width: `${(8 - this.state.dataToShow.length % 8) * 12.5}%` }}></div>
            )}
          </div>
        </Modal>
        <Row id="scatterRow">
          <Col span={24} id="scatter" className="scatterChart">
            <div style={{ display: 'flex' }}>
              {/* 侧边栏 */}
              <div style={{ width: '20px', marginRight: '20px', textAlign: 'center' }}>
                <ul style={{ listStyleType: 'none', padding: 0 }}>
                  {[0, 1, 2, 3].map(index => (
                    <li key={index} style={{marginBottom: '20px' }}>
                     <Button style={{ width: '20px',height:'60px', backgroundColor: 'lightblue' ,writingMode: 'vertical-lr', textOrientation: 'mixed',color :'black' }} onClick={() => this.handleWSIClick(index)}>WSI-{index + 1}</Button>
                    </li>
                  ))}
                </ul>
              </div>
              {/* 右侧部分 */}
              <div>
                <Row gutter={[16, 16]}> {/* 使用 gutter 定义列之间的间距 */}
                  <Col span={8} style={{ marginRight: '15px' }}>
                    <Button
                      onClick={this.newFunction}
                      style={{
                        backgroundColor: '#1890ff', // 背景色
                        color: '#fff', // 字体颜色
                        border: 'none', // 去除边框
                        borderRadius: '4px', // 圆角
                        padding: '10px 20px', // 内边距
                        boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.1)', // 阴影
                        transition: 'background-color 0.3s, transform 0.3s', // 过渡效果
                      }}
                      onMouseEnter={(e) => { e.target.style.backgroundColor = '#40a9ff'; }} // 悬停效果
                      onMouseLeave={(e) => { e.target.style.backgroundColor = '#1890ff'; }} // 悬停效果
                    >
                      <PlusOutlined style={{ marginRight: '4px' }} /> {/* 添加Ant Design的图标 */}
                      该图片噪声点列表
                    </Button>
                  </Col>
                  <Col span={15} style={{ marginLeft: '6px' }}>
                    <Row gutter={[8, 8]}> {/* 使用gutter添加行和列之间的间距 */}
                      <Button
                        onClick={() => this.newFunction2('无癌症')}
                        style={{ fontSize: '12px', padding: '8px 16px', borderRadius: '20px',marginRight: '20px',backgroundColor:"#40b373",color: '#fff' }} // 添加圆角和更大的内边距
                      >
                        无癌症切片
                      </Button>
                      <Button 
                        onClick={() => this.newFunction2('有癌症')}
                        style={{ fontSize: '12px', padding: '8px 16px', borderRadius: '20px' , marginRight: '20px',backgroundColor:"#d4b446",color: '#fff'}} // 添加圆角和更大的内边距
                      >
                        有癌症切片
                      </Button>
                      <Button
                        onClick={() => this.newFunction2('高癌症')}
                        style={{ fontSize: '12px', padding: '8px 16px', borderRadius: '20px',backgroundColor:"#ee8826",color: '#fff' }} // 添加圆角和更大的内边距
                      >
                        高癌症切片
                      </Button>
                    </Row>
                  </Col>
                </Row>
                <Row > {/* 使用 gutter 定义列之间的间距 */}
                  <Col span={13}>
                    <Row
                      style={{
                        width: '250px', // 设置较小的宽度
                        minHeight: '50px', // 设置最小高度
                        maxHeight: '500px', // 设置最大高度
                        overflowY: 'auto', // 设置垂直滚动
                        marginTop: '10px',
                        marginLeft: '20px', // 添加右边距
                      }}
                    >
                      <ul style={{ listStyleType: 'none', padding: '15px 0', margin: 0 }}>
                        {this.state.noiseChild.map((item, index) => (
                          <li key={index} style={{ marginBottom: '12px' }}>{item}</li>
                        ))}
                      </ul>
                    </Row>
                  </Col>
                  <Col span={10}>
                    <Row
                      style={{
                        width: '250px', // 设置较小的宽度
                        height: '500px', // 设置固定高度
                        overflowY: 'auto', // 设置垂直滚动
                        marginTop: '10px',
                      }}
                    >
                      <ul style={{ listStyleType: 'none', padding: '10px 0', margin: 0 }}>
                        {this.state.dataToShow.map((item, index) => (
                          <li key={index} style={{ marginBottom: '10px', color: this.state.datacolor }}>{item}</li>
                        ))}
                      </ul>
                    </Row>
                  </Col>
                </Row>
              </div>
            </div>
          </Col>
          <Affix style={{ position: 'fixed', bottom: 300, right: 300 }}>
            <Button onClick = {this.openDraw}type="primary" shape="rect">修改噪声切片标签</Button>
          </Affix>
          <Affix style={{ position: 'fixed', bottom: 300, right: 30 }}>
            <Button onClick = {this.showDetails}type="primary" shape="rect">查看细节</Button>
          </Affix>
        </Row>



        <Modal
          title="噪声切片修改页面"
          visible={this.state.visible}
          onOk={this.handleOk}
          confirmLoading={this.state.confirmLoading}
          onCancel={this.handleCancel}
          width={800}

        >
          <Col style={{ marginBottom: '10px' }}>
            <Text style={{ marginTop: '10px' }}>
              WSI 图片编号 : WSI-{this.state.selectedWSI}
            </Text>
          </Col>
          <Row style={{ marginBottom: '30px' }}>
            <Row>
              <Text style={{ marginRight: '10px' }}>
                选择噪声切片:
              </Text>
              <Select
                value={this.state.selectedOption}
                onChange={this.handleSelectChange}
                style={{ width: '100%' }}
              >
                {this.state.noiseChild.map((option, index) => (
                  <Select.Option key={index} value={option}>
                    {option}
                  </Select.Option>
                ))}
              </Select>
            </Row>
          </Row>
          <Row gutter={[0, 1]} justify="space-around" align="small">
            <Col span={10} offset={2}>
                <Image style = "small"
                  fallback="data:image/png;base64,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"
                  src={process.env.REACT_APP_IMAGE_PATH_NEW + "/image_" + this.state.selectedWSI + "/" + this.state.result}
                  alt={`Selected Image`}
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
                {/* <div>{process.env.REACT_APP_IMAGE_PATH_NEW + "/image_" + this.state.selectedWSI + "/" + this.state.result}</div> */}
            </Col>
            {/* <Col span={10} offset={1}>
              <Text type="secondary">Grad-Cam Image</Text>
              <Image
                fallback="data:image/png;base64,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"
                width={160}
                height={160}
                src={
                  "./data/init_data_image" +
                  "/image_CAM_" +
                  this.props.chooseMapImg +
                  "/" +
                  this.state.index2 +
                  ".png"
                }

              />
            </Col> */}
            <Col span={24}>
              <Descriptions layout="vertical" size="small" column={4} bordered>
                <Descriptions.Item span={2} label="参考值:Bmm和PR分数">
                  <ReactECharts
                    style={{ width: 190, height: 80 }}
                    option={this.state.barOption}
                  />
                </Descriptions.Item>
                <Descriptions.Item span={2} label="参考值:Bmm和PR迭代变化趋势">
                  <ReactECharts
                    style={{ width: 200, height: 110 }}
                    option={this.state.areaOption}
                  />
                </Descriptions.Item>
                <Descriptions.Item span={4} label="目前标签">
                  
                 
                    <div className="demo-option-label-item">
                      {this.state.recentclass === 1 && <Tag color={category_color[0]}>无癌症</Tag>}
                      {this.state.recentclass === 2 && <Tag color={category_color[1]}>有癌症</Tag>}
                      {this.state.recentclass === 3 && <Tag color={category_color[2]}>高癌症</Tag>}
                    </div>
                  
                
                </Descriptions.Item>
                <Descriptions.Item span={4} label="更新目标标签">
                  <Select
                    style={{ width: "100%" }}
                    placeholder="选择一个标签"
                    optionLabelProp="label"
                    onChange={(value) => this.handleSelectChange_2(value)}
                  >
                    {category_name.map((item, index) => {
                      return (
                        <Option value={index} label={item}>
                          <div className="demo-option-label-item">
                            <Tag color={category_color[index]}>{item}</Tag>
                          </div>
                        </Option>
                      );
                    })}
                  </Select>
                </Descriptions.Item>
              </Descriptions>
            </Col>
          </Row>

        </Modal>
      </>
    );
  }
}
