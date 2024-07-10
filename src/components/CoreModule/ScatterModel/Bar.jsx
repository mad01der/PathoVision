/*
 * @Date: 2022-07-21 11:08:42
 * @LastEditors: JZY
 * @LastEditTime: 2022-12-17 13:28:31
 * @FilePath: /project/Visual/src/components/CoreModule/ScatterModel/Bar.jsx
 */
import React, { Component } from "react";
import ReactECharts from "echarts-for-react";
import { message , Typography ,Descriptions,Button } from "antd";
import axios from "axios";
import * as d3 from "d3";
const { Text } = Typography;
const category_name = ["LUSC", "LUAD"];
const category_color = ["blue", "geekblue", "purple"]; // and more
const noise_tag_color = ['#90b7b7', '#a9906c', '#ef9a9a']
const noise_tag_name = ["clean", "noise", "high-noise"];
export default class Bar extends Component {
  constructor(props) {
    super(props);

    this.state = {
      index: [],
      infor_state : [],
      infor_patch_id :[],
      data_len :0,
      Bmm: [],
      Spr: [],
      option: {},
      choosePatches: props.choosePatches,
    };
  }
  infor = (a,b,c,d) =>{
    axios
    .post('http://127.0.0.1:5000/delete', {
     a,
     b,
     c,
     d
     })
     .then(response => {
       console.log('Response from server:', response.data);
     })
     .catch(error => {
       console.error('Error:', error);
     });
     window.location.reload()
  }
  componentDidMount = () => {
    this.props.onChildEvent(this);
    this.drawChart(0, -1);
  };
  check_new_result = () => {
    this.props.onChildEvent(this);
    this.drawChart(0, -1);
  }
  changeBarRange = (id) => {
    this.drawChart(id, id);
  };
  // selectBar = {
  //   click: async (e) => {
  //     const newTags = this.state.choosePatches.filter(
  //       (tag) => tag !== e.dataIndex
  //     );
  //     if (newTags.length < 10) {
  //       await newTags.push(e.dataIndex);
  //       await this.setState({
  //         choosePatches: newTags,
  //       });
  //     } else {
  //       await message.error("The selected image has reached the limitation!");
  //     }
  //     this.props.changeChoosePatches(newTags);
  //   },
  // };
  getInfor = () => {
    var infor = [];
    var data_len = Object.keys(this.props.New_data.patch_id).length;
    console.log(" _",data_len);
    for(var i = 0;i<data_len;i++){
      infor.push([
        this.props.New_data.patch_id[i],
        this.props.New_data.img_id[i],
        this.props.New_data.selected_label[i],
        this.props.New_data.origin_label[i],
      ]);
    }
    this.setState({
      infor_state : infor,
      data_len:data_len
    });
  }
  drawChart = (start, end) => {
    this.getInfor();
    var index = [];
    var patchNum = [];
    var Bmm = [];
    var Spr = [];
    var img_len = Object.keys(this.props.WSI_Data.img_id).length;
    for (var i = 0; i < img_len; i++) {
      index.push(this.props.WSI_Data["img_id"][i]);
      patchNum.push(this.props.WSI_Data["patch_num"][i]);
      Bmm.push(this.props.WSI_Data["bmm"][i]);
      Spr.push(this.props.WSI_Data["spr"][i]);
    }
    // normalize
    function normalize(arr) {
      let max = 0;
      let min=2;
      let t=0.005;
      arr.forEach(v => {
        max=Math.max(v,max);
        min=Math.min(min,v);
      })
      for (let i = 0; i < arr.length; i++) {
        arr[i] = (arr[i]+t-min) / (max-min);
      }
      return arr
    }

    if (end == -1) end = index.length - 1;
    this.setState({
      option: {
        title: {
          text: "Noise Metric for Images:",
        },
        legend: {
          top: 10,
          textStyle: {
            fontSize: 10,
          },
        },
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "shadow",
          },
        },
        grid: {
          top: 30,
          bottom: 50,
          left: 30,
          right: 0,
        },
        dataZoom: [
          {
            type: "slider",
            xAxisIndex: 0,
            left: 15,
            bottom: 0,
            startValue: start,
            endValue: end,
          },
          {
            type: "inside",
          },

        ],
        xAxis: {
          data: index,
          silent: false,
          splitLine: {
            show: false,
          },
          splitArea: {
            show: false,
          },
        },
        yAxis: [
          {
            position: "left",
            type: "value",
            min: 0,
            max: 1.05,
            splitArea: {
              show: false,
            },
          },
        ],
        series: [
          {
            name: "Bmm Score",
            type: "bar",
            data: Bmm,
            large: true,
          },
          {
            name: "Spr Score",
            type: "bar",
            data: normalize(Spr),
            large: true,
          },
      
        ],
      },
    });

  };

  render() {
    return (
      <>
         <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: '#000' }}>Edit records</h2>
        {/* <ReactECharts
          style={{ height: "23vh" }}
          option={this.state.option}
          onEvents={this.selectBar}
        /> */}
        <div style={{ overflowX: 'auto', maxHeight: '200px' }}>
          {[...Array(this.state.data_len).keys()].reverse().map(index => (
            <div key={index} style={{ display: 'flex' }}>
              <div style={{ flex: 1, border: '4px solid #cfc', padding: '5px', marginRight: '10px', position: 'relative' }}>
                
                <div>WSI ID: {this.props.New_data.img_id[index]}</div>
                <div>slice ID: {this.props.New_data.patch_id[index]}</div>
                <div>origin_label: {this.props.New_data.origin_label[index]}</div>
                <div>edit_label: {this.props.New_data.selected_label[index]}</div>
                <Button onClick={() => this.infor(this.props.New_data.img_id[index],this.props.New_data.patch_id[index],this.props.New_data.origin_label[index],this.props.New_data.selected_label[index])} style={{
                  position: 'absolute', top: '5px', right: '5px',
                  backgroundColor: '#ff0000', // 背景色
                  color: '#fff', // 字体颜色
                  border: 'none', // 去除边框
                  borderRadius: '4px',
                }}>Undo</Button>
              </div>
            </div>
  ))}
        </div>
      </>
    );
  }
}
