import React, { Component } from 'react';
import { Space, Table, Tag ,Button,Col} from 'antd';
import type { TableProps } from 'antd';
import { Link } from 'react-router-dom';
import './Data.css'; // Import your custom CSS file
import background from "./background.jpg";
import axios from "axios";
interface DataType {
  key: string;
  name: string;
  age: string;
  tags: string[];
}

interface Props {}
interface State {}
export default class Data extends Component<Props, State> {
  columns: TableProps<DataType>['columns'];
  data: DataType[];

  constructor(props: Props) {
    super(props);
    this.state = {
     currentDate: new Date(),
     columns :[
        {
          title: '就诊号',
          dataIndex: 'name',
          key: 'name',
          render: (text: string) => <a>{text}</a>,
        },
        {
          title: '报告号',
          dataIndex: 'age',
          key: 'age',
        },
        {
          title: '日期',
          dataIndex: 'time',
          key: 'time',
        },
        {
          title: '检查结果',
          key: 'tags',
          dataIndex: 'tags',
          render: (_, { tags }) => (
            <>
              {tags.map((tag) => {
                let color = tag.length > 5 ? 'geekblue' : 'red';
                if (tag === '无癌症') {
                  color = 'blue';
                }
                return (
                  <Tag color={color} key={tag}>
                    {tag.toUpperCase()}
                  </Tag>
                );
              })}
            </>
          ),
        },
        {
          title: '操作',
          key: 'action',
          render: (_, record) => (
            <>
              <Button type="primary" style={{ marginRight: '10px' }} onClick={() => this.jump(record.key)}>
                查看
              </Button>
              <Button type="primary" style={{ backgroundColor: 'red', borderColor: 'red' }} onClick={() => this.delete(record.key)}>
                删除
              </Button>
            </>
          ),
        },
      ],
      data : [
        {
          key: '1',
          name: '20240415110',
          age:  '20240415110-0001',
          time: '2024/4/15',
          tags: ['报告生成完成'],
        },
       
      ]
    };
  }
  
  jump = (record) =>{
    if(record == 1){
         this.props.history.replace('/Home_load');
    }
    if(record > 1){
      this.props.history.replace('/Home');
     }
  }
  jump2 = () =>{
    this.props.history.replace('/Load');
  }
  jump3 = () =>{
    this.props.history.replace('/Login');
  }
  delete = (value) =>{
     console.log(value - 1);
     axios.post('http://127.0.0.1:5000/mysql3', { ID: value - 1})
        .then(response => {
            // 检查是否删除成功
            if (response.data.delete_status === 200) {
                console.log('Value deleted successfully:', value);
            } else {
                console.error('Failed to delete value:', value);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
     window.location.reload();
  }
  componentDidMount() {
    axios.get('http://127.0.0.1:5000/mysql2')
        .then(response => {
            // 检查是否请求成功
            if (response.data.load_status === 200) {
                // 遍历数据列表，更新前端状态
                const newData = response.data.data.map(patient => ({
                    key: String(patient.patient_key + 1), // 使用患者ID作为新数据的 key
                    name: patient.patient_id,
                    age: patient.patient_name,
                    time: new Date().toLocaleDateString(), // 假设使用当前日期作为时间
                    tags: ['报告生成完成'], // 示例中假设新增的数据标记为 '报告生成完成'
                }));
                // 更新前端状态中的数据列表，将后端返回的数据放在固定数据的后面
                this.setState({
                    data: [{  // 固定的第一行数据
                      key: '1',
                      name: '20240415110',
                      age:  '20240415110-0001',
                      time: '2024/4/15',
                      tags: ['报告生成完成'],
                    }, ...newData] // 将后端返回的数据放在后面
                });
            } else {
                console.error('Failed to load data:', response.data.load_status);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}
  render() {
    return (
      <div className="page-container" style={{ backgroundImage: `url(${background})`, opacity: 0.9 }}>
        <div className="return-button-container" style={{ position: 'absolute', top: '50px', left: '50px' }}>
              <Button type="primary" onClick={this.jump3} >返回登录页面</Button>
        </div>
        <Col>
          <Col>
            <Button className="button-container" type = "primary" onClick = {this.jump2}> 新增</Button>
          </Col>
          <Col className="table-container">
            <Table columns={this.state.columns} dataSource={this.state.data} />
          </Col>
        </Col>
      </div>
    );
  }
}
