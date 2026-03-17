// RUN: xdsl-opt -p vectorize-reduction{vector_size=4} %s | filecheck %s

// CHECK:       builtin.module {

// -- Float32 reduction -------------------------------------------------------
// CHECK-NEXT:    func.func @sum_f32(%{{.*}} : memref<16xf32>, %{{.*}} : memref<f32>) {
func.func @sum_f32(%input: memref<16xf32>, %init: memref<f32>) {
  linalg.reduce ins(%input : memref<16xf32>) outs(%init : memref<f32>)
                dimensions = [0]
  (%in : f32, %acc : f32) {
    %s = arith.addf %in, %acc : f32
    linalg.yield %s : f32
  }
  return
}
// CHECK-NEXT:      %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:      %{{.*}} = arith.constant 4 : index
// CHECK-NEXT:      %{{.*}} = arith.constant 16 : index
// CHECK-NEXT:      %{{.*}} = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-NEXT:      %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<4xf32>) {
// CHECK-NEXT:        %{{.*}} = vector.load %{{.*}}[%{{.*}}] : memref<16xf32>, vector<4xf32>
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-NEXT:        scf.yield %{{.*}} : vector<4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %{{.*}} = vector.reduction <add>, %{{.*}} : vector<4xf32> into f32
// CHECK-NEXT:      %{{.*}} = memref.load %{{.*}}[] : memref<f32>
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:      memref.store %{{.*}}, %{{.*}}[] : memref<f32>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }


// -- Integer (i32) reduction -------------------------------------------------
// CHECK-NEXT:    func.func @sum_i32(%{{.*}} : memref<8xi32>, %{{.*}} : memref<i32>) {
func.func @sum_i32(%input: memref<8xi32>, %init: memref<i32>) {
  linalg.reduce ins(%input : memref<8xi32>) outs(%init : memref<i32>)
                dimensions = [0]
  (%in : i32, %acc : i32) {
    %s = arith.addi %in, %acc : i32
    linalg.yield %s : i32
  }
  return
}
// CHECK-NEXT:      %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:      %{{.*}} = arith.constant 4 : index
// CHECK-NEXT:      %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:      %{{.*}} = arith.constant dense<0> : vector<4xi32>
// CHECK-NEXT:      %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<4xi32>) {
// CHECK-NEXT:        %{{.*}} = vector.load %{{.*}}[%{{.*}}] : memref<8xi32>, vector<4xi32>
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : vector<4xi32>
// CHECK-NEXT:        scf.yield %{{.*}} : vector<4xi32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %{{.*}} = vector.reduction <add>, %{{.*}} : vector<4xi32> into i32
// CHECK-NEXT:      %{{.*}} = memref.load %{{.*}}[] : memref<i32>
// CHECK-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      memref.store %{{.*}}, %{{.*}}[] : memref<i32>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// CHECK-NEXT:  }
